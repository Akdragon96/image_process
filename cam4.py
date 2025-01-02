import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Check the device (CPU or GPU)
print(f"Model is running on: {model.device}")

cap = cv2.VideoCapture('cars_video.mp4')
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            # Draw rectangle
            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print(conf)
            # Class name
            cls = box.cls[0]
            cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

