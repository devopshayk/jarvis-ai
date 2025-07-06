# JARVIS AI - Hand Recognition and Voice Assistant

A Python-based AI assistant that combines hand gesture recognition, voice interaction, and computer vision for electronic component identification and board detection.

## Features

- **Hand Gesture Recognition**: Interactive hand tracking with pinch gestures for object manipulation
- **Voice Assistant**: Speech-to-text and text-to-speech capabilities with ChatGPT integration
- **Board Detection**: Computer vision-based detection of electronic boards and components
- **Image Search**: Automatic image search and display for electronic components
- **Real-time Processing**: Live camera feed with gesture and voice interaction

## Files

- `Hand_Recognition.py`: Hand gesture tracking and object manipulation
- `jarvis_board_detector_pro.py`: Electronic board detection using computer vision
- `new_speech_to_text.py`: Main voice assistant with ChatGPT integration
- `test_voice_search.py`: Image search functionality for electronic components

## Setup

### Prerequisites

Install the required Python packages:

```bash
pip install opencv-python mediapipe numpy openai google-cloud-speech edge-tts pyaudio requests pillow ultralytics
```

### API Keys Setup

This project requires several API keys. Set them as environment variables:

```bash
# OpenAI API Key (for ChatGPT integration)
export OPENAI_API_KEY="your_openai_api_key_here"

# ElevenLabs API Key (for text-to-speech)
export ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"

# Google Cloud Speech API credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"
```

### Getting API Keys

1. **OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com/) and create an API key
2. **ElevenLabs API Key**: Sign up at [ElevenLabs](https://elevenlabs.io/) and get your API key
3. **Google Cloud Speech**: Set up a Google Cloud project and download your service account credentials

## Usage

### Running the Voice Assistant

```bash
python new_speech_to_text.py
```

### Running Hand Recognition

```bash
python Hand_Recognition.py
```

### Running Board Detection

```bash
python jarvis_board_detector_pro.py
```

## Controls

### Voice Commands
- Say "Hey Jarvis" to wake up the assistant
- Ask questions about electronic components
- Request images of specific components
- Use board detection commands

### Hand Gestures
- Pinch gestures for object manipulation
- Drag, resize, and rotate objects
- Delete objects by dragging to trash zone

### Keyboard Controls
- `q`: Quit the application
- `1`, `2`: Switch between detection modes
- `ESC`: Exit
- `r`: Reset crop box
- `d`: Toggle debug mode
- `c`: Toggle calibration mode

## Security Notes

- API keys are stored as environment variables for security
- No sensitive data is hardcoded in the source files
- The `.gitignore` file prevents accidental commit of sensitive files

## Troubleshooting

1. **Camera not working**: Make sure your webcam is connected and accessible
2. **Audio issues**: Check your microphone and speakers are properly configured
3. **API errors**: Verify your API keys are correctly set as environment variables
4. **Permission errors**: Ensure you have the necessary permissions for camera and audio access

## License

This project is for educational and personal use. Please respect the terms of service for the APIs used. 