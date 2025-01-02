import cv2
import math
import torch
import cvzone
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Force the model to use GPU if available
#device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    gpu_name=torch.cuda.get_device_name(torch.cuda.current_device())
    device="cuda"
    print(f"gpu in use : {gpu_name}")
else:
    print("No gpu detected, running on cpu")
    device="cpu"
model.to(device)
print(f"Model is running on: {model.device}")
classnames=["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
            "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
            "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
            "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
            "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
            "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
            "teddy bear","hair dryer","toothbrush"
            ]
cap = cv2.VideoCapture(0)
#cap=cv2.VideoCapture('cars_video.mp4')
#cap=cv2.VideoCapture('pedersian.mp4')
#cap=cv2.VideoCapture('cars_mask.mp4')
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img=cv2.flip(img,1)
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
            cls = int(box.cls[0])
            currentclass=classnames[cls]
            #cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
            cvzone.putTextRect(img,f'{currentclass} {conf}',(max(0,x1),max(35,y1)))
            """if currentclass=="truck" and conf>0.3:
                cvzone.putTextRect(img,f'{currentclass} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                cvzone.cornerRect(img,(x1,y1,w,h),l=9)"""
    cv2.imshow("Image", img)
    cv2.waitKey(1)

