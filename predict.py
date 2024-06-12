import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolov8n.pt")  # pretrained YOLOv8n model

# Read an image using OpenCV
cvImg = cv2.imread("ultralytics/assets/bus.jpg")

# Run inference on the source
results = model(cvImg, imgsz=640, conf=0.2)  # list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    print(list(boxes.cls))
    print(list(boxes.conf))
    print(list(boxes.xywhn))