import os
import asyncio
import edge_tts
import openai
import pyaudio
from google.cloud import speech
import subprocess
import cv2
import math
import mediapipe as mp
import numpy as np
from Hand_Recognition import HandTracker, OverlayObject
import threading
import requests
from PIL import Image
from io import BytesIO
from test_voice_search import ImageSearcher
from ultralytics import YOLO
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"     # Fix OpenCV GUI on Wayland

# Global HandTracker instance
images_list = []
DEFAULT_IMAGE_DIR = './searched_images/'

# Global screen center coordinates (will be updated by camera_loop)
screen_center_x = 320
screen_center_y = 240

# Create default image directory if it doesn't exist
if not os.path.exists(DEFAULT_IMAGE_DIR):
    os.makedirs(DEFAULT_IMAGE_DIR)
    print(f"Created directory: {DEFAULT_IMAGE_DIR}")

# Dictionary to track image paths for objects
object_image_paths = {}

# Global board detector instance
board_detector = None
detection_enabled = False  # Start with detection disabled
current_frame = None  # Store the current camera frame for analysis

# Board detector integration
class JarvisBoardDetector:
    def __init__(self):
        self.mode = 1  # 1: Image Detection (only mode now)
        self.mode_names = {1: 'Image Detection'}
        # Removed YOLO model and COCO classes since we're only using image detection

    def image_detection(self, img):
        """Enhanced detection for microschemas and electronic boards"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple detection methods for better coverage
        detections = []
        
        # Method 1: Original adaptive threshold (good for larger boards)
        blur1 = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)
        kernel1 = np.ones((7,7), np.uint8)
        closed1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel1)
        contours1, _ = cv2.findContours(closed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 2: Canny edge detection (good for microschemas)
        blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur2, 50, 150)
        kernel2 = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel2, iterations=1)
        contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 3: Multi-scale detection (for different sizes)
        all_contours = []
        all_contours.extend(contours1)
        all_contours.extend(contours2)
        
        for scale in [0.5, 1.0, 2.0]:
            if scale != 1.0:
                scaled_img = cv2.resize(gray, None, fx=scale, fy=scale)
            else:
                scaled_img = gray
                
            blur3 = cv2.GaussianBlur(scaled_img, (3, 3), 0)
            thresh3 = cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel3 = np.ones((3,3), np.uint8)
            closed3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel3)
            contours3, _ = cv2.findContours(closed3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Scale contours back to original size
            if scale != 1.0:
                for cnt in contours3:
                    cnt = cnt.astype(np.float32)
                    cnt[:, :, 0] = cnt[:, :, 0] / scale
                    cnt[:, :, 1] = cnt[:, :, 1] / scale
                    cnt = cnt.astype(np.int32)
            all_contours.extend(contours3)
        
        # Enhanced filtering for electronic components
        for cnt in all_contours:
            area = cv2.contourArea(cnt)
            
            # Lower minimum area for microschemas (was 1000, now 100)
            if area > 100:
                # Get bounding rectangle
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # Calculate aspect ratio
                width = rect[1][0]
                height = rect[1][1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                
                # Check if it looks like an electronic component
                # Electronic boards typically have aspect ratios between 0.5 and 3.0
                if 0.5 <= aspect_ratio <= 3.0:
                    # Calculate perimeter to area ratio (helps identify complex shapes like circuits)
                    perimeter = cv2.arcLength(cnt, True)
                    perimeter_area_ratio = perimeter / area if area > 0 else 0
                    
                    # Electronic components often have higher perimeter/area ratios
                    if perimeter_area_ratio > 0.1:  # Threshold for complex shapes
                        detections.append((rect, area, perimeter_area_ratio))
        
        # Return the best detection (highest area with good characteristics)
        if detections:
            # Sort by area and perimeter/area ratio
            detections.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return detections[0][0]  # Return the best rect
        
        return None

    def draw_mode_buttons(self, frame):
        h, w = frame.shape[:2]
        button_texts = ['Image Detection Active', 'q: Quit']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        margin = 20
        button_height = 40
        button_widths = [cv2.getTextSize(t, font, font_scale, thickness)[0][0] + 30 for t in button_texts]
        total_width = sum(button_widths) + margin * (len(button_texts) - 1)
        x = w - total_width - 20
        y = 20
        for i, text in enumerate(button_texts):
            bw = button_widths[i]
            color = (0, 255, 0) if i == 0 else (50, 50, 50)  # First button is always active
            cv2.rectangle(frame, (x, y), (x + bw, y + button_height), color, -1)
            cv2.putText(frame, text, (x + 15, y + button_height - 12), font, font_scale, (0, 0, 0), thickness)
            x += bw + margin

    def draw_detected_labels(self, frame, labels):
        if not labels:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        margin = 10
        labels = [str(label) for label in labels]
        label_sizes = [cv2.getTextSize(label, font, font_scale, thickness)[0] for label in labels]
        label_height = sum([size[1] for size in label_sizes]) + margin * (len(labels) + 1)
        label_width = max([size[0] for size in label_sizes]) + 2 * margin
        cv2.rectangle(frame, (10, 10), (10 + label_width, 10 + label_height), (0, 255, 0), -1)
        y = 10 + margin
        for i, label in enumerate(labels):
            cv2.putText(frame, label, (20, y + label_sizes[i][1]), font, font_scale, (0, 0, 0), thickness)
            y += label_sizes[i][1] + margin

    def process_frame(self, frame):
        """Process a single frame and return the processed frame with detections"""
        found = False
        detected_labels = []
        
        # Only use image detection now
        rect = self.image_detection(frame)
        if rect:
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            cv2.drawContours(frame, [box], 0, (0,255,0), 3)
            x, y = int(np.min(box[:,0])), int(np.min(box[:,1]))
            label = 'Electronic Board'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            found = True
        
        # Add mode indicator
        cv2.putText(frame, f'Mode: {self.mode_names[self.mode]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        if not found:
            cv2.putText(frame, 'No electronic board detected', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        self.draw_mode_buttons(frame)
        
        return frame, found, detected_labels

def get_openai_response(model, messages, temperature=None):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature or 0.7,
        max_tokens=150
    )
    return response.choices[0].message.content

def on_object_deleted(deleted_path):
    global images_list
    if deleted_path in images_list:
        images_list.remove(deleted_path)
        print(f"Removed {deleted_path} from images list")

def update_handtracker_images(new_images):
    global hand_tracker
    if hand_tracker:
        hand_tracker.objects = []
        for i, path in enumerate(new_images):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Failed to load {path}")
                img = np.zeros((100, 100, 4), dtype=np.uint8)
                img[..., :3] = 0
                img[..., 3] = 128
            elif img.shape[2] == 3:
                alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
                img = np.dstack((img, alpha))
            
            offset_x = i * 20
            offset_y = i * 20
            obj = OverlayObject(img, position=(screen_center_x + offset_x, screen_center_y + offset_y), image_path=path)
            hand_tracker.objects.append(obj)
            hand_tracker.reset_object_original_size(obj)

class SpeechToSpeech:
    def __init__(self):
        # API Keys - Use environment variables for security
        # Set these environment variables before running:
        # export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
        # export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
        # export OPENAI_API_KEY="your_openai_api_key"
        
        # Get API keys from environment variables
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY', 'your_elevenlabs_api_key_here')
        openai.api_key = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
        
        # Check if required API keys are set
        if not openai.api_key or openai.api_key == 'your_openai_api_key_here':
            print("Warning: OPENAI_API_KEY environment variable not set")
        if not self.elevenlabs_api_key or self.elevenlabs_api_key == 'your_elevenlabs_api_key_here':
            print("Warning: ELEVENLABS_API_KEY environment variable not set")
            
        self.conversation_history = [
            {"role": "system", "content": "You are an expert engineering assistant. Help with tech and describe image requests if needed."}
        ]
        self.sample_rate = 16000
        self.block_size = 1024
        self.silence_threshold = 3
        self.silence_duration = 1.0

    def ask_chatgpt(self, prompt):
        self.conversation_history.append({"role": "user", "content": prompt + '\n If respond will be more than 100 word plaese make max 100 w'})
        lines = self.detect_and_describe_image_request(prompt)
        if "yes" in lines.lower():
            lines = lines.split('\n')
            obj = lines[1].split(":")[1].strip()
            desc = lines[2].split(":")[1].strip()
            return f"You asked for a picture of a {obj}. Here's what I found:\n{desc}"
        else:
            reply = get_openai_response("gpt-4", self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": reply})
            
            # Only show component images if the user specifically asked for pictures
            if self.detect_picture_request(prompt):
                components_found = self.extract_electronic_components(reply)
                if components_found:
                    return reply + "\n\n" + self.auto_show_component_images(components_found)
            
            return reply

    def detect_picture_request(self, user_input):
        """Detect if user specifically asked for pictures or images"""
        user_input_lower = user_input.lower()
        
        picture_keywords = [
            'picture', 'pictures', 'image', 'images', 'photo', 'photos',
            'show me', 'show pictures', 'show images', 'display',
            'what does it look like', 'how does it look', 'visual',
            'see', 'show', 'picture of', 'image of'
        ]
        
        for keyword in picture_keywords:
            if keyword in user_input_lower:
                return True
        return False

    def extract_electronic_components(self, text):
        """Extract electronic component names from text"""
        # Common electronic components
        component_keywords = [
            'resistor', 'capacitor', 'transistor', 'diode', 'led', 'ic', 'integrated circuit',
            'microcontroller', 'arduino', 'raspberry pi', 'breadboard', 'wire', 'battery',
            'switch', 'button', 'potentiometer', 'variable resistor', 'relay', 'transformer',
            'inductor', 'crystal', 'oscillator', 'sensor', 'display', 'lcd', 'seven segment',
            'motor', 'servo', 'stepper', 'fan', 'heatsink', 'pcb', 'circuit board',
            'timer', '555', 'op amp', 'operational amplifier', 'voltage regulator',
            'power supply', 'connector', 'header', 'jumper', 'cable', 'multimeter',
            'oscilloscope', 'soldering iron', 'solder', 'flux', 'schematic'
        ]
        
        found_components = []
        text_lower = text.lower()
        
        for component in component_keywords:
            if component in text_lower:
                found_components.append(component)
        
        return found_components

    def auto_show_component_images(self, components):
        """Automatically search for and display component images"""
        if not components:
            return ""
        
        found_images = []
        descriptions = []
        
        for component in components:
            print(f"Auto-searching for image of: {component}")
            
            # Search for the component image
            safe_component_name = component.replace(' ', '_').replace('/', '_').lower()
            local_filename = os.path.join(DEFAULT_IMAGE_DIR, f"{safe_component_name}.png")
            
            if not os.path.exists(local_filename):
                # Try alternative path from searched_images
                searched_filename = os.path.join('searched_images', f"{safe_component_name}.png")
                if os.path.exists(searched_filename):
                    local_filename = searched_filename
                else:
                    # Try to download the image
                    print(f"Auto-downloading image for: '{component}'")
                    searcher = ImageSearcher(component)
                    searcher.run()
                    
                    # Check if the image was successfully downloaded
                    if not os.path.exists(local_filename):
                        # Try alternative path from searched_images
                        searched_filename = os.path.join('searched_images', f"{safe_component_name}.png")
                        if os.path.exists(searched_filename):
                            local_filename = searched_filename
            
            # Add to images list if found
            if os.path.exists(local_filename):
                if local_filename not in images_list:
                    images_list.append(local_filename)
                    print(f"Auto-added image for '{component}': {local_filename}")
                
                found_images.append(local_filename)
                
                # Get component description
                component_desc = self.get_component_description(component)
                descriptions.append(f"{component}: {component_desc}")
            else:
                print(f"Could not find image for '{component}'")
        
        # Update the hand tracker with all found images
        if found_images:
            update_handtracker_images(images_list)
            return f"\nI've automatically found and displayed images for: {', '.join(components)}.\n" + "\n".join(descriptions)
        
        return ""

    def get_component_description(self, component):
        """Get a brief description of an electronic component"""
        descriptions = {
            'resistor': 'Limits current flow in circuits',
            'capacitor': 'Stores electrical charge temporarily',
            'transistor': 'Amplifies or switches electronic signals',
            'diode': 'Allows current flow in one direction only',
            'led': 'Light-emitting diode for indicators',
            'ic': 'Integrated circuit with multiple functions',
            'integrated circuit': 'Complex electronic circuit on a chip',
            'microcontroller': 'Programmable computer on a chip',
            'arduino': 'Popular microcontroller development platform',
            'breadboard': 'Temporary circuit construction board',
            'wire': 'Conducts electricity between components',
            'battery': 'Provides electrical power',
            'switch': 'Opens or closes electrical circuit',
            'button': 'Momentary switch for input',
            'potentiometer': 'Variable resistor for volume/control',
            'timer': '555 timer IC for timing circuits',
            '555': 'Popular timer integrated circuit',
            'op amp': 'Operational amplifier for signal processing',
            'voltage regulator': 'Maintains constant voltage output',
            'pcb': 'Printed circuit board for mounting components',
            'circuit board': 'Board with conductive tracks for circuits',
            'sensor': 'Detects physical properties (light, temperature, etc.)',
            'display': 'Shows information visually',
            'lcd': 'Liquid crystal display for text/graphics',
            'motor': 'Converts electrical energy to mechanical motion',
            'servo': 'Position-controlled motor',
            'multimeter': 'Measures voltage, current, and resistance',
            'soldering iron': 'Tool for joining electronic components',
            'schematic': 'Diagram showing circuit connections'
        }
        
        return descriptions.get(component, 'Electronic component')

    async def speak(self, text):
        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")
        try:
            communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and "data" in chunk:
                    with open("temp.mp3", "ab") as f:
                        f.write(chunk["data"])
            os.system("ffplay -nodisp -autoexit temp.mp3")
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to a different male voice
            try:
                communicate = edge_tts.Communicate(text, voice="en-US-TonyNeural")
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and "data" in chunk:
                        with open("temp.mp3", "ab") as f:
                            f.write(chunk["data"])
                os.system("ffplay -nodisp -autoexit temp.mp3")
            except Exception as e2:
                print(f"Fallback TTS Error: {e2}")
                # Final fallback to female voice if male voices fail
                communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio" and "data" in chunk:
                        with open("temp.mp3", "ab") as f:
                            f.write(chunk["data"])
                os.system("ffplay -nodisp -autoexit temp.mp3")
        
    def start_transcription(self):
        RATE = 16000
        CHUNK = int(RATE / 10)  # 100ms
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        
        def mic_stream():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            try:
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    yield data
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

        requests_iterator = speech.StreamingRecognizeRequest(audio_content=content)
        responses = client.streaming_recognize(config, requests_iterator)
        
        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if result.is_final:
                return transcript
        return None

    def detect_and_describe_image_request(self, user_input):
        """Detect if user is asking for image description"""
        user_input_lower = user_input.lower()
        
        image_request_keywords = [
            'describe', 'what is this', 'what do you see', 'identify',
            'what kind of', 'what type of', 'tell me about this',
            'analyze', 'examine', 'look at this', 'see this'
        ]
        
        for keyword in image_request_keywords:
            if keyword in user_input_lower:
                return "yes\nobject: electronic component\ndescription: This appears to be an electronic component or circuit board"
        return "no"

    def detect_board_detector_commands(self, user_input):
        """Detect commands related to board detection"""
        user_input_lower = user_input.lower()
        return any(keyword in user_input_lower for keyword in ['detect', 'board', 'circuit', 'component'])

    def detect_identification_request(self, user_input):
        """Detect if user wants to identify something"""
        user_input_lower = user_input.lower()
        
        identification_keywords = [
            'identify', 'what is this', 'what kind of', 'what type of',
            'recognize', 'detect', 'analyze', 'examine', 'look at',
            'tell me what', 'describe this', 'what do you see'
        ]
        
        for keyword in identification_keywords:
            if keyword in user_input_lower:
                return True
        return False

    def analyze_current_frame(self):
        """Analyze the current camera frame for electronic components"""
        global current_frame, board_detector
        
        if current_frame is None:
            return "No camera frame available for analysis."
        
        if board_detector is None:
            board_detector = JarvisBoardDetector()
        
        try:
            # Process the frame
            processed_frame, found, detected_labels = board_detector.process_frame(current_frame.copy())
            
            if found:
                analysis = "I can see an electronic board or component in the camera view. "
                analysis += "The detection system has identified it as an electronic board. "
                analysis += "This could be a circuit board, microcontroller, or other electronic component."
                
                # Add specific details if available
                if detected_labels:
                    analysis += f" Detected labels: {', '.join(detected_labels)}"
                
                return analysis
            else:
                return "I don't see any electronic boards or components in the current camera view. Try positioning the camera closer to the object or ensure good lighting."
                
        except Exception as e:
            return f"Error analyzing the frame: {str(e)}"

    async def wait_for_wake_up(self):
        """Wait for wake-up command"""
        print("🎤 Waiting for wake-up command: 'Hey Jarvis'")
        while True:
            try:
                transcript = self.start_transcription()
                if transcript and "hey jarvis" in transcript.lower():
                    print("🎯 Wake-up command detected!")
                    await self.speak("Hello! I'm ready to help you.")
                    return True
            except Exception as e:
                print(f"Error in wake-up detection: {e}")
                await asyncio.sleep(0.1)

    async def start_conversation(self):
        # Wait for wake-up command first
        await self.wait_for_wake_up()
        
        print("🗣️ Starting conversation mode...")
        await self.speak("I'm ready to help you with your questions. What would you like to know?")
        
        while True:
            try:
                print("🎤 Listening...")
                transcript = self.start_transcription()
                
                if transcript:
                    print(f"👤 You said: {transcript}")
                    
                    # Check for quit command
                    if any(word in transcript.lower() for word in ['quit', 'exit', 'stop', 'goodbye']):
                        await self.speak("Goodbye! Have a great day!")
                        break
                    
                    # Check for board detection commands
                    if self.detect_board_detector_commands(transcript):
                        await self.speak("I'll analyze the current camera view for electronic components.")
                        analysis = self.analyze_current_frame()
                        await self.speak(analysis)
                        continue
                    
                    # Check for identification requests
                    if self.detect_identification_request(transcript):
                        await self.speak("I'll analyze what I can see in the camera view.")
                        analysis = self.analyze_current_frame()
                        await self.speak(analysis)
                        continue
                    
                    # Get ChatGPT response
                    response = self.ask_chatgpt(transcript)
                    print(f"🤖 Assistant: {response}")
                    await self.speak(response)
                    
            except Exception as e:
                print(f"Error in conversation: {e}")
                await asyncio.sleep(0.1)

def play_sound(filename):
    try:
        os.system(f"ffplay -nodisp -autoexit {filename}")
    except:
        pass

def camera_loop():
    global hand_tracker, current_frame, detection_enabled
    
    # Initialize hand tracker with empty images list
    hand_tracker = HandTracker(images_list, on_object_deleted)
    hand_tracker.on_object_deleted = on_object_deleted

    print("📹 Starting camera loop...")
    while True:
        try:
            ret, frame = hand_tracker.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_frame = frame.copy()  # Store current frame for analysis
            
            # Process frame with hand tracking
            processed_frame = hand_tracker.process(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r'):  # 'r' key to reset crop box
                hand_tracker.reset_crop_box()
            elif key == ord('d'):  # 'd' key to toggle debug mode
                hand_tracker.debug_mode = not hand_tracker.debug_mode
            elif key == ord('c'):  # 'c' key to toggle calibration mode
                hand_tracker.calibration_mode = not hand_tracker.calibration_mode
            
            cv2.imshow('JARVIS AI - Hand Recognition', processed_frame)
            
        except Exception as e:
            print(f"Error in camera loop: {e}")
            break
    
    hand_tracker.cap.release()
    cv2.destroyAllWindows()

async def voice_loop():
    sts = SpeechToSpeech()
    await sts.start_conversation()

async def main():
    # Run voice loop in main event loop (camera will start after wake-up)
    await voice_loop()

if __name__ == "__main__":
    asyncio.run(main())
