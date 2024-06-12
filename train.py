from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml").load("weights/yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="ultralytics/cfg/datasets/medical.yaml", epochs=100, imgsz=640, degrees=180, batch=0.8)