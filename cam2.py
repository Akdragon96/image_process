import cv2
from ultralytics import YOLO
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    results=model(frame,show=True)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
