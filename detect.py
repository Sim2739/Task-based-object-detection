from ultralytics import YOLO
import cv2

# Load YOLOv5n via ultralytics
model = YOLO('yolov5n.pt')

image_path = 'image.jpg'

results = model(image_path)

# Print detections
for r in results:
    boxes = r.boxes
    for box in boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf)
        print(f"Detected: {label}, Confidence: {confidence:.2f}")

# Show image with boxes drawn
results[0].show()