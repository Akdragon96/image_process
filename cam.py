from ultralytics import YOLO
import cv2
import cvzone 
import math
#cap=cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")
#classnames=["person","bicycle","car"]
results=model("car.jpg",show=True)
cv2.waitKey(0)
