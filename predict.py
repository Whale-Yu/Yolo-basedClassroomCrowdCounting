from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("weights/best-yolov8n.pt")

# Run inference on 'bus.jpg' with arguments
results = model('HeadDatasets300/test/images/20241018_105056_frame_011.jpg', imgsz=640, conf=0.05, iou=0.3, save=True, project="outputs")
