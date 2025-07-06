import cv2
import math
import mediapipe as mp
from mediapipe.python.solutions import hands as mphands
import numpy as np
import time


class UI:
    def __init__(self):
        self.buttons = {
            'debug': {'pos': (60, 20), 'size': (90, 32), 'text': 'Debug', 'color': (0, 130, 200)},
            'calibrate': {'pos': (170, 20), 'size': (110, 32), 'text': 'Calibrate', 'color': (230, 25, 75)},
            'addimg': {'pos': (300, 20), 'size': (110, 32), 'text': 'Add Img', 'color': (60, 180, 75)}
        }
        self.button_states = {
            'debug': False,
            'calibrate': False
        }

    def draw_button(self, image, button_name, button_info):
        x, y = button_info['pos']
        w, h = button_info['size']
        color = button_info['color']
        text = button_info['text']
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    def draw_ui(self, image, pinch_types=None):
        for button_name, button_info in self.buttons.items():
            self.draw_button(image, button_name, button_info)
        import datetime
        now = datetime.datetime.now().strftime('%H:%M:%S')
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(now, font, font_scale, thickness)[0]
        img_h, img_w = image.shape[:2]
        text_x = (img_w - text_size[0]) // 2
        text_y = 18
        cv2.putText(image, now, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        if pinch_types and len(pinch_types) > 0:
            pinch_type = pinch_types[0][0].upper()
            pinch_text = f'Pinch: {pinch_type}'
            pinch_text_size = cv2.getTextSize(pinch_text, font, 0.7, 2)[0]
            pinch_text_x = (img_w - pinch_text_size[0]) // 2
            pinch_text_y = text_y + 28
            cv2.putText(image, pinch_text, (pinch_text_x, pinch_text_y), font, 0.7, (255, 255, 255), 2)

    def check_button_hover(self, x, y):
        for button_name, button_info in self.buttons.items():
            bx, by = button_info['pos']
            bw, bh = button_info['size']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button_name
        return None


class OverlayObject:
    def __init__(self, image, position=(200, 200), size=300, angle=0, image_path=None):
        self.image = image
        self.position = list(position)
        self.size = size
        self.original_size = size
        self.angle = angle
        self.dragging = False
        self.drag_hand_id = None
        self.offset = [0, 0]
        self.last_scale_distance = None
        self.scale_anchor = None
        self.last_angle = None
        self.rotation_anchor = None
        self.image_path = image_path
        
        self.is_animating = False
        self.animation_start_time = 0
        self.animation_duration = 0.3
        self.animation_start_size = size
        self.animation_target_size = size
        self.pinch_scale_factor = 1.2
        
        self.is_in_trash_zone = False
        self.trash_scale_factor = 0.5
        self.trash_animation_duration = 0.2
        
        self.target_position = list(position)
        self.drag_smoothness = 0.3
        self.last_valid_position = list(position)
        
        self.target_size = size


class HandTracker:
    def __init__(self, images, on_object_deleted=None):
        self.hands = mphands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.on_object_deleted = on_object_deleted
        
        self.ui = UI()
        
        self.cap = cv2.VideoCapture(1)
        self.crop_box = None
        self.debug_mode = False
        self.calibration_mode = False
        self.mpHands = mphands

        image_paths = images

        self.objects = []
        for i, path in enumerate(image_paths):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Failed to load {path}")
                img = np.zeros((100, 100, 4), dtype=np.uint8)
                img[..., :3] = 0
                img[..., 3] = 128
            elif img.shape[2] == 3:
                alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
                img = np.dstack((img, alpha))
            screen_center_x = 320
            screen_center_y = 240
            
            offset_x = i * 20
            offset_y = i * 20
            obj = OverlayObject(img, position=(screen_center_x + offset_x, screen_center_y + offset_y), image_path=path)
            self.objects.append(obj)
            self.reset_object_original_size(obj)

        self.active_object = None
        self.previous_pinch_states = {}

    def detect_and_crop_orange_area(self, frame):
        return frame

    def reset_crop_box(self):
        self.crop_box = None

    def calibrate_orange_detection(self, frame):
        if not self.calibration_mode:
            return frame
            
        cal_frame = frame.copy()
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        hsv_center = hsv[center_y, center_x]
        

        
        cv2.line(cal_frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 255), 2)
        cv2.line(cal_frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 255), 2)
        
        return cal_frame

    def start_pinch_animation(self, obj, is_pinching):
        current_time = time.time()
        obj.is_animating = True
        obj.animation_start_time = current_time
        obj.animation_start_size = obj.size
        
        if is_pinching:
            if obj.is_in_trash_zone:
                obj.animation_target_size = int(obj.original_size * obj.trash_scale_factor * obj.pinch_scale_factor)
            else:
                obj.animation_target_size = int(obj.original_size * obj.pinch_scale_factor)
        else:
            if obj.is_in_trash_zone:
                obj.animation_target_size = int(obj.original_size * obj.trash_scale_factor)
            else:
                obj.animation_target_size = obj.original_size

    def start_trash_zone_animation(self, obj, entering_trash_zone, is_pinched=False):
        current_time = time.time()
        obj.is_animating = True
        obj.animation_start_time = current_time
        obj.animation_start_size = obj.size
        
        if entering_trash_zone:
            if is_pinched:
                obj.animation_target_size = int(obj.original_size * obj.trash_scale_factor * obj.pinch_scale_factor)
            else:
                obj.animation_target_size = int(obj.original_size * obj.trash_scale_factor)
            obj.animation_duration = obj.trash_animation_duration
        else:
            if is_pinched:
                obj.animation_target_size = int(obj.original_size * obj.pinch_scale_factor)
            else:
                obj.animation_target_size = obj.original_size
            obj.animation_duration = 0.3

    def update_animations(self):
        current_time = time.time()
        
        for obj in self.objects:
            if obj.is_animating:
                elapsed = current_time - obj.animation_start_time
                progress = min(elapsed / obj.animation_duration, 1.0)
                
                ease_progress = 1 - (1 - progress) ** 3
                
                start_size = obj.animation_start_size
                target_size = obj.animation_target_size
                obj.size = int(start_size + (target_size - start_size) * ease_progress)
                
                if progress >= 1.0:
                    obj.is_animating = False
                    obj.size = obj.animation_target_size
            
            obj.size += (obj.target_size - obj.size) * 0.25
            
            if obj.dragging:
                obj.position[0] += (obj.target_position[0] - obj.position[0]) * obj.drag_smoothness
                obj.position[1] += (obj.target_position[1] - obj.position[1]) * obj.drag_smoothness
            else:
                obj.position[0] += (obj.target_position[0] - obj.position[0]) * 0.1
                obj.position[1] += (obj.target_position[1] - obj.position[1]) * 0.1

    def reset_object_original_size(self, obj):
        obj.original_size = obj.size
        obj.animation_target_size = obj.size

    def is_inside_square(self, point, center, size):
        x, y = point
        cx, cy = center
        return (cx - size // 2 < x < cx + size // 2) and (cy - size // 2 < y < cy + size // 2)

    def get_pinch_type(self, lm, w, h, threshold=25):
        x_thumb, y_thumb = int(lm[4].x * w), int(lm[4].y * h)
        x_index, y_index = int(lm[8].x * w), int(lm[8].y * h)
        x_middle, y_middle = int(lm[12].x * w), int(lm[12].y * h)
        
        d_index = math.hypot(x_index - x_thumb, y_index - y_thumb)
        d_middle = math.hypot(x_middle - x_thumb, y_middle - y_thumb)
        
        if d_index < threshold and d_middle < threshold:
            if d_index < d_middle:
                return 'index', ((x_thumb + x_index) // 2, (y_thumb + y_index) // 2)
            else:
                return 'middle', ((x_thumb + x_middle) // 2, (y_thumb + y_middle) // 2)
        elif d_index < threshold:
            return 'index', ((x_thumb + x_index) // 2, (y_thumb + y_index) // 2)
        elif d_middle < threshold:
            return 'middle', ((x_thumb + x_middle) // 2, (y_thumb + y_middle) // 2)
        else:
            return None, None

    def detect_connected_fingers(self, lm, w, h):
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        middle_tip = (int(lm[12].x * w), int(lm[12].y * h))
        ring_tip = (int(lm[16].x * w), int(lm[16].y * h))
        pinky_tip = (int(lm[20].x * w), int(lm[20].y * h))
        
        index_middle_dist = math.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])
        middle_ring_dist = math.hypot(middle_tip[0] - ring_tip[0], middle_tip[1] - ring_tip[1])
        ring_pinky_dist = math.hypot(ring_tip[0] - pinky_tip[0], ring_tip[1] - pinky_tip[1])
        
        # Check if fingers are connected (close to each other)
        connection_threshold = 30  # Distance threshold for "connected" fingers
        fingers_connected = (index_middle_dist < connection_threshold and 
                           middle_ring_dist < connection_threshold and 
                           ring_pinky_dist < connection_threshold)
        
        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
        thumb_fingers_dist = math.hypot(thumb_tip[0] - middle_tip[0], thumb_tip[1] - middle_tip[1])
        
        return fingers_connected, thumb_fingers_dist

    def get_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_angle(self, p1, p2):
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    def expand_canvas_for_rotation(self, overlay):
        h, w = overlay.shape[:2]
        diag = int(math.ceil(math.sqrt(h**2 + w**2)))
        if overlay.shape[2] == 4:
            canvas = np.zeros((diag, diag, 4), dtype=overlay.dtype)
        else:
            canvas = np.zeros((diag, diag, 3), dtype=overlay.dtype)
        x = (diag - w) // 2
        y = (diag - h) // 2
        canvas[y:y+h, x:x+w] = overlay
        return canvas

    def overlay_transparent(self, background, overlay, x, y, overlay_size=None, angle=0):
        h, w = overlay.shape[:2]
        if overlay_size is not None:
            overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

        overlay = self.expand_canvas_for_rotation(overlay)
        overlay_rotated = cv2.warpAffine(
            overlay,
            cv2.getRotationMatrix2D((overlay.shape[1] // 2, overlay.shape[0] // 2), angle, 1.0),
            (overlay.shape[1], overlay.shape[0])
        )

        # After rotating:
        h, w = overlay_rotated.shape[:2]
        x_draw = int(x - w // 2)
        y_draw = int(y - h // 2)

        # Compute overlay and background region coordinates (clipping to image bounds)
        b_h, b_w = background.shape[:2]
        x1, y1 = max(x_draw, 0), max(y_draw, 0)
        x2, y2 = min(x_draw + w, b_w), min(y_draw + h, b_h)
        ox1, oy1 = max(0, -x_draw), max(0, -y_draw)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if x1 >= x2 or y1 >= y2:
            return background

        overlay_img = overlay_rotated[oy1:oy2, ox1:ox2, :3]
        mask = overlay_rotated[oy1:oy2, ox1:ox2, 3:] / 255.0
        # Make the overlay a little transparent (alpha multiplier)
        alpha_multiplier = 0.7
        mask = mask * alpha_multiplier
        background_slice = background[y1:y2, x1:x2]
        blended = background_slice * (1 - mask) + overlay_img * mask
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        return background

    def handle_gesture_logic_drag_resize_rotate(self, obj, pinch_types, connected_finger_hands, w, h):
        # pinch_types: list of (type, center, hand_index)
        # connected_finger_hands: list of (landmarks, hand_index, thumb_fingers_dist) for scaling
        # Single hand pinch (either type): drag
        # One hand pinching + one hand with connected fingers: resize with connected fingers hand
        # Both hands 'index': rotate
        # Any other: only drag if one hand
        if len(pinch_types) == 1 and len(connected_finger_hands) == 1:
            # One hand pinching (drag) + one hand with connected fingers (scale)
            _, pinch_center, i = pinch_types[0]
            lm_scale, _, thumb_fingers_dist = connected_finger_hands[0]
            
            # Handle dragging
            obj.last_angle = None
            obj.rotation_anchor = None
            if not obj.dragging and self.is_inside_square(pinch_center, obj.position, obj.size):
                obj.dragging = True
                obj.drag_hand_id = i
                obj.offset = [pinch_center[0] - obj.position[0], pinch_center[1] - obj.position[1]]
                obj.last_valid_position = obj.position.copy()
                # Move the object to the top layer when dragging starts
                if obj in self.objects:
                    self.objects.remove(obj)
                    self.objects.append(obj)
            if obj.dragging and i == obj.drag_hand_id:
                # Update target position for smooth dragging
                obj.target_position[0] = pinch_center[0] - obj.offset[0]
                obj.target_position[1] = pinch_center[1] - obj.offset[1]
                obj.last_valid_position = obj.target_position.copy()
            
            # Handle scaling with connected fingers hand
            # Use distance between thumb and the connected fingers group
            obj.last_scale_distance = thumb_fingers_dist
            
        elif len(pinch_types) == 1:
            # Only one hand pinching - just drag
            _, pinch_center, i = pinch_types[0]
            obj.last_scale_distance = None
            obj.scale_anchor = None
            obj.last_angle = None
            obj.rotation_anchor = None
            if not obj.dragging and self.is_inside_square(pinch_center, obj.position, obj.size):
                obj.dragging = True
                obj.drag_hand_id = i
                obj.offset = [pinch_center[0] - obj.position[0], pinch_center[1] - obj.position[1]]
                obj.last_valid_position = obj.position.copy()
                # Move the object to the top layer when dragging starts
                if obj in self.objects:
                    self.objects.remove(obj)
                    self.objects.append(obj)
            if obj.dragging and i == obj.drag_hand_id:
                # Update target position for smooth dragging
                obj.target_position[0] = pinch_center[0] - obj.offset[0]
                obj.target_position[1] = pinch_center[1] - obj.offset[1]
                obj.last_valid_position = obj.target_position.copy()
        elif len(pinch_types) == 2:
            t0, c0, _ = pinch_types[0]
            t1, c1, _ = pinch_types[1]
            if t0 == 'index' and t1 == 'index':
                # Rotate
                current_angle = self.get_angle(c0, c1)
                if obj.last_angle is not None:
                    angle_diff = current_angle - obj.last_angle
                    obj.angle += angle_diff
                    # Snap to 90 deg
                    snap_interval = 90
                    snap_threshold = 10
                    nearest_snap = round(obj.angle / snap_interval) * snap_interval
                    if abs(obj.angle - nearest_snap) < snap_threshold:
                        obj.angle = nearest_snap
                obj.last_angle = current_angle
                obj.last_scale_distance = None
                obj.dragging = False
            else:
                # Any other combination: do nothing
                obj.last_scale_distance = None
                obj.scale_anchor = None
                obj.last_angle = None
                obj.rotation_anchor = None
                obj.dragging = False
        else:
            # No gesture
            obj.last_scale_distance = None
            obj.scale_anchor = None
            obj.last_angle = None
            obj.rotation_anchor = None
            obj.dragging = False

    def process(self, image):
        h, w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Use hand_landmarks for all logic instead of results.multi_hand_landmarks
        hand_landmarks = getattr(results, 'multi_hand_landmarks', None)
        pinch_types = []  # List of ('index' or 'middle', center, hand_index)
        connected_finger_hands = []  # List of hands with connected fingers for scaling
        if hand_landmarks:
            for i, hand in enumerate(hand_landmarks):
                lm = hand.landmark
                pinch_type, pinch_center = self.get_pinch_type(lm, w, h)
                if pinch_type:
                    pinch_types.append((pinch_type, pinch_center, i))
                else:
                    # Check if hand has connected fingers for scaling
                    fingers_connected, thumb_fingers_dist = self.detect_connected_fingers(lm, w, h)
                    if fingers_connected:
                        connected_finger_hands.append((lm, i, thumb_fingers_dist))

        # --- Minimalistic UI button activation with pinch (index or middle) ---
        # Allow button actions if an index or middle finger pinch is detected over a button
        if len(pinch_types) > 0:
            for pinch_type, pinch_center, _ in pinch_types:
                if pinch_type in ('index', 'middle'):
                    button_name = self.ui.check_button_hover(*pinch_center)
                    if button_name == 'debug':
                        self.ui.button_states['debug'] = not self.ui.button_states['debug']
                        self.debug_mode = self.ui.button_states['debug']
                    elif button_name == 'calibrate':
                        self.ui.button_states['calibrate'] = not self.ui.button_states['calibrate']
                        self.calibration_mode = self.ui.button_states['calibrate']
                    elif button_name == 'addimg':
                        # Add images.png as a new object
                        img = cv2.imread('images.png', cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            if img.shape[2] == 3:
                                alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
                                img = np.dstack((img, alpha))
                            screen_center_x = w // 2
                            screen_center_y = h // 2
                            obj = OverlayObject(img, position=(screen_center_x, screen_center_y), image_path='images.png')
                            self.objects.append(obj)
                            self.reset_object_original_size(obj)

        # Draw UI overlay (minimalistic)
        self.ui.draw_ui(image, pinch_types)

        # Draw hand exoskeleton (landmarks and connections)
        if hand_landmarks:
            for hand in hand_landmarks:
                self.mpDraw.draw_landmarks(image, hand, mphands.HAND_CONNECTIONS)

        # Draw points at pinch locations
        for pinch_type, pinch_center, _ in pinch_types:
            color = (0, 255, 0) if pinch_type == 'index' else (255, 0, 0)
            cv2.circle(image, pinch_center, 12, color, -1)

        # Check if active object is in delete zone and no longer being pinched
        delete_zone_size = 120  # Increased size for better usability
        object_in_delete_zone = False
        object_to_delete = None

        # Check trash zone state for all objects
        for obj in self.objects:
            obj_x, obj_y = obj.position
            obj_size = obj.size
            
            # Check if center of object is in delete zone (more precise)
            in_trash_zone = obj_x < delete_zone_size and obj_y > h - delete_zone_size
            
            # Handle trash zone animation
            if in_trash_zone != obj.is_in_trash_zone:
                # Trash zone state changed
                # Check if this object is currently being pinched
                is_pinched = False
                if len(pinch_types) > 0:
                    for _, pinch_center, _ in pinch_types:
                        if pinch_center and self.is_inside_square(pinch_center, obj.position, obj.size):
                            is_pinched = True
                            break
                self.start_trash_zone_animation(obj, in_trash_zone, is_pinched)
                obj.is_in_trash_zone = in_trash_zone

            
            if in_trash_zone:
                object_in_delete_zone = True
                # If no pinch is active, mark this object for deletion
                if len(pinch_types) == 0:
                    object_to_delete = obj
                    break

        # Track pinch state changes and trigger animations
        current_pinch_states = {}
        
        # First, determine which object is actively being pinched (if any)
        active_pinched_object = None
        if len(pinch_types) > 0:
            for obj in reversed(self.objects):  # Check from top to bottom (topmost first)
                for _, pinch_center, _ in pinch_types:
                    if pinch_center and self.is_inside_square(pinch_center, obj.position, obj.size):
                        active_pinched_object = obj
                        break
                if active_pinched_object:
                    break
        
        # Now update pinch states - only the active object should be considered pinched
        for obj in self.objects:
            is_pinched = (obj == active_pinched_object)
            current_pinch_states[obj] = is_pinched
            
            # Check if pinch state changed
            previous_state = self.previous_pinch_states.get(obj, False)
            if is_pinched != previous_state:
                # Pinch state changed, start animation
                self.start_pinch_animation(obj, is_pinched)
        
        # Update previous states
        self.previous_pinch_states = current_pinch_states

        # Improved single-object interaction logic
        if self.active_object is not None:
            # If there are no pinches, clear the active object
            if len(pinch_types) == 0:
                self.active_object.dragging = False
                self.active_object.drag_hand_id = None
                self.active_object.last_scale_distance = None
                self.active_object.scale_anchor = None
                self.active_object.last_angle = None
                self.active_object.rotation_anchor = None
                self.active_object = None
        if self.active_object is None and len(pinch_types) > 0:
            # Find the topmost object under a pinch to activate
            for obj in reversed(self.objects):  # Check from top to bottom
                if any(self.is_inside_square(center, obj.position, obj.size) for _, center, _ in pinch_types if center):
                    self.active_object = obj
                    break

        # Only allow the active object to be modified, reset others
        for obj in self.objects:
            if obj != self.active_object:
                obj.dragging = False
                obj.drag_hand_id = None
                obj.last_scale_distance = None
                obj.scale_anchor = None
                obj.last_angle = None
                obj.rotation_anchor = None

        if self.active_object:
            self.handle_gesture_logic_drag_resize_rotate(self.active_object, pinch_types, connected_finger_hands, w, h)

        # --- Two-hand C-shape scaling gesture (robust) ---
        # If both hands have connected fingers (C shape), scale the topmost object between them
        if len(connected_finger_hands) == 2:
            lm1, _, _ = connected_finger_hands[0]
            lm2, _, _ = connected_finger_hands[1]
            # Use middle finger tip as C center
            c1 = (int(lm1[12].x * w), int(lm1[12].y * h))
            c2 = (int(lm2[12].x * w), int(lm2[12].y * h))
            current_distance = min(self.get_distance(c1, c2), 300)
            # Draw debug line and distance
            cv2.line(image, c1, c2, (255, 255, 0), 2)
            mid_x = (c1[0] + c2[0]) // 2
            mid_y = (c1[1] + c2[1]) // 2
            cv2.putText(image, f'{int(current_distance)}', (mid_x, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            # Find the topmost object between the two hands
            if self.objects:
                obj = self.objects[-1]  # Topmost object
                # Track initial distance and size
                if not hasattr(self, '_cshape_scale_active') or not self._cshape_scale_active:
                    self._cshape_scale_active = True
                    self._cshape_initial_distance = current_distance
                    self._cshape_initial_size = obj.size
                else:
                    scale_factor = current_distance / max(1, self._cshape_initial_distance)
                    new_size = int(self._cshape_initial_size * scale_factor)
                    obj.target_size = max(50, min(new_size, 500))
                    obj.original_size = obj.target_size
        else:
            self._cshape_scale_active = False

        # Update all animations
        self.update_animations()

        # Delete the object if it's in the delete zone and no pinch is active
        if object_to_delete:
            print(f"DELETING OBJECT at position ({object_to_delete.position[0]}, {object_to_delete.position[1]})")
            deleted_path = object_to_delete.image_path
            self.objects.remove(object_to_delete)
            if self.active_object == object_to_delete:
                self.active_object = None
            object_in_delete_zone = False
            
            # Call the callback function if provided
            if self.on_object_deleted and deleted_path:
                self.on_object_deleted(deleted_path)

        # Draw realistic trash bin in bottom-left corner
        bin_width = 80
        bin_height = 60
        bin_x = 30
        bin_y = h - delete_zone_size + 30
        
        # Main trash bin background with gradient effect
        if object_in_delete_zone:
            # Glow effect when object is in delete zone
            for i in range(3):
                glow_color = (0, 0, 100 - i * 30)
                cv2.rectangle(image, (bin_x - i*2, bin_y - i*2), (bin_x + bin_width + i*2, bin_y + bin_height + i*2), glow_color, -1)
        else:
            # Normal trash bin background
            cv2.rectangle(image, (bin_x, bin_y), (bin_x + bin_width, bin_y + bin_height), (40, 40, 40), -1)
        
        # Trash bin body (metallic gray)
        cv2.rectangle(image, (bin_x, bin_y), (bin_x + bin_width, bin_y + bin_height), (80, 80, 80), -1)
        cv2.rectangle(image, (bin_x, bin_y), (bin_x + bin_width, bin_y + bin_height), (120, 120, 120), 2)
        
        # Trash bin lid (darker gray with handle)
        lid_height = 15
        cv2.rectangle(image, (bin_x - 8, bin_y - lid_height), (bin_x + bin_width + 8, bin_y), (60, 60, 60), -1)
        cv2.rectangle(image, (bin_x - 8, bin_y - lid_height), (bin_x + bin_width + 8, bin_y), (100, 100, 100), 2)
        
        # Lid handle
        handle_width = 20
        handle_height = 8
        handle_x = bin_x + (bin_width - handle_width) // 2
        handle_y = bin_y - lid_height - handle_height
        cv2.rectangle(image, (handle_x, handle_y), (handle_x + handle_width, handle_y + handle_height), (40, 40, 40), -1)
        cv2.rectangle(image, (handle_x, handle_y), (handle_x + handle_width, handle_y + handle_height), (80, 80, 80), 2)
        
        # Trash bin opening (black interior)
        opening_margin = 8
        cv2.rectangle(image, (bin_x + opening_margin, bin_y + opening_margin), 
                     (bin_x + bin_width - opening_margin, bin_y + bin_height - opening_margin), (20, 20, 20), -1)
        
        # Trash bin texture lines (horizontal)
        for i in range(4):
            line_y = bin_y + 12 + i * 10
            if line_y < bin_y + bin_height - opening_margin:
                cv2.line(image, (bin_x + opening_margin, line_y), (bin_x + bin_width - opening_margin, line_y), (60, 60, 60), 1)
        
        # Trash bin texture lines (vertical)
        for i in range(3):
            line_x = bin_x + 15 + i * 20
            if line_x < bin_x + bin_width - opening_margin:
                cv2.line(image, (line_x, bin_y + opening_margin), (line_x, bin_y + bin_height - opening_margin), (60, 60, 60), 1)
        
        # Trash bin feet/supports
        foot_width = 8
        foot_height = 4
        for i in range(3):
            foot_x = bin_x + 10 + i * 25
            cv2.rectangle(image, (foot_x, bin_y + bin_height), (foot_x + foot_width, bin_y + bin_height + foot_height), (40, 40, 40), -1)
        
        # Add subtle shadow
        shadow_offset = 3
        cv2.rectangle(image, (bin_x + shadow_offset, bin_y + shadow_offset), 
                     (bin_x + bin_width + shadow_offset, bin_y + bin_height + shadow_offset), (20, 20, 20), -1)

        for obj in self.objects:
            h0, w0 = obj.image.shape[:2]
            aspect = h0 / w0
            new_w = int(obj.size)
            new_h = int(obj.size * aspect)
            image = self.overlay_transparent(
                image, obj.image, obj.position[0], obj.position[1], (new_w, new_h), -int(obj.angle)
            )

        return image


# Main loop to use the tracker
if __name__ == "__main__":
    tracker = HandTracker(["images.png", "pngtree-blue-science-fiction-grid-iron-man-ai-jarvis-png-image_2962692-removebg-preview.png"])
    
    # Setup for optimal camera view
    tracker.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    tracker.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker.cap.set(cv2.CAP_PROP_FPS, 30)
    tracker.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)
    tracker.cap.set(cv2.CAP_PROP_CONTRAST, 0.3)
    tracker.cap.set(cv2.CAP_PROP_SATURATION, 0.3)
    tracker.cap.set(cv2.CAP_PROP_HUE, 0.0)
    tracker.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    while True:
        ret, frame = tracker.cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        
        # Calibration mode takes priority
        if tracker.calibration_mode:
            frame = tracker.calibrate_orange_detection(frame)
        else:
            # Detect and crop orange area
            frame = tracker.detect_and_crop_orange_area(frame)
        
        output = tracker.process(frame)
        

        
        # Create windowed mode (not fullscreen)
        cv2.namedWindow("Gesture Overlay", cv2.WINDOW_NORMAL)

        # Get the frame dimensions
        frame_height, frame_width = output.shape[:2]

        # Set a reasonable window size that fits most screens
        window_width = 1280
        window_height = 720

        # Calculate scale to fit the frame inside the window while preserving aspect ratio
        scale = min(window_width / frame_width, window_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Resize the frame
        resized_output = cv2.resize(output, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create a black background
        display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        # Compute top-left corner for centering
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2
        # Paste the resized frame onto the black background
        display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_output

        # Set window size
        cv2.resizeWindow("Gesture Overlay", window_width, window_height)
        # Position window at center of screen
        cv2.moveWindow("Gesture Overlay", 100, 100)

        cv2.imshow("Gesture Overlay", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # 'r' key to reset crop box
            tracker.reset_crop_box()
        elif key == ord('d'):  # 'd' key to toggle debug mode
            tracker.debug_mode = not tracker.debug_mode
        elif key == ord('c'):  # 'c' key to toggle calibration mode
            tracker.calibration_mode = not tracker.calibration_mode

    tracker.cap.release()
    cv2.destroyAllWindows()
