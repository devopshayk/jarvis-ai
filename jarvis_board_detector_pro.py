import cv2
import numpy as np
from ultralytics import YOLO

# Paths to models
YOLO_GENERAL_MODEL = 'yolov8n.pt'  # General object detection (COCO dataset)

# Classic image processing board detection
def image_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)
    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            area = cv2.contourArea(box)
            if area > max_area:
                max_area = area
                best_rect = rect
    return best_rect

def draw_yolo_detections(frame, results, class_names):
    found = False
    detected_labels = set()
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            found = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Always use class name, fallback to 'Unknown' if not found
                class_name = class_names[cls] if (isinstance(class_names, list) and cls < len(class_names)) else 'Unknown'
                detected_labels.add(class_name)
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw object name INSIDE the box at the top left
                label = f'{class_name}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                label_bg_y2 = y1 + th + 10
                cv2.rectangle(frame, (x1, y1), (x1 + tw, label_bg_y2), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                # Draw confidence at the bottom of the box
                conf_label = f'{conf:.2f}'
                (ctw, cth), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y2 - cth - 10), (x1 + ctw, y2), (0, 255, 0), -1)
                cv2.putText(frame, conf_label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return found, sorted(list(detected_labels))

def draw_mode_buttons(frame, mode, mode_names):
    h, w = frame.shape[:2]
    button_texts = [f'1: {mode_names[1]}', f'2: {mode_names[2]}', 'q: Quit']
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
        color = (0, 255, 0) if (i + 1 == mode) else (50, 50, 50)
        cv2.rectangle(frame, (x, y), (x + bw, y + button_height), color, -1)
        cv2.putText(frame, text, (x + 15, y + button_height - 12), font, font_scale, (0, 0, 0), thickness)
        x += bw + margin

def draw_detected_labels(frame, labels):
    if not labels:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    margin = 10
    label_height = 0
    label_width = 0
    # Ensure all labels are strings
    labels = [str(label) for label in labels]
    label_sizes = [cv2.getTextSize(label, font, font_scale, thickness)[0] for label in labels]
    label_height = sum([size[1] for size in label_sizes]) + margin * (len(labels) + 1)
    label_width = max([size[0] for size in label_sizes]) + 2 * margin
    # Draw background rectangle
    cv2.rectangle(frame, (10, 10), (10 + label_width, 10 + label_height), (0, 255, 0), -1)
    y = 10 + margin
    for i, label in enumerate(labels):
        cv2.putText(frame, label, (20, y + label_sizes[i][1]), font, font_scale, (0, 0, 0), thickness)
        y += label_sizes[i][1] + margin

def main():
    # Load YOLO model (COCO dataset)
    yolo_general = YOLO(YOLO_GENERAL_MODEL)
    general_class_names = list(getattr(getattr(yolo_general, 'model', None), 'names', [str(i) for i in range(80)]))

    mode = 1  # 1: General YOLO, 2: Image Detection
    mode_names = {1: 'General YOLO', 2: 'Image Detection'}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ Could not open webcam')
        return
    print('🚀 JARVIS Board Detector Pro - Press 1: General YOLO, 2: Image Detection, q: Quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        found = False
        detected_labels = []
        if mode == 1:
            results = yolo_general(frame, conf=0.25, verbose=False)
            found, detected_labels = draw_yolo_detections(frame, results, general_class_names)
        elif mode == 2:
            rect = image_detection(frame)
            if rect:
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                cv2.drawContours(frame, [box], 0, (0,255,0), 3)
                # Draw 'Electronic Board' label at the top of the box
                x, y = int(np.min(box[:,0])), int(np.min(box[:,1]))
                label = 'Electronic Board'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                found = True
        # UI
        cv2.putText(frame, f'Mode: {mode_names[mode]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        if not found:
            cv2.putText(frame, 'No object/board detected', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        draw_mode_buttons(frame, mode, mode_names)
        if mode == 1:
            draw_detected_labels(frame, detected_labels)
        cv2.imshow('JARVIS Board Detector Pro', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
    cap.release()
    cv2.destroyAllWindows()
    print('✅ Detection stopped')

if __name__ == '__main__':
    main() 