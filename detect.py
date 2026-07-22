"""
Run YOLOv8 detection on validation images.
Uses the latest trained model (welding_defect_model6).
"""

from ultralytics import YOLO

# Use the latest trained model
MODEL_PATH = "runs/detect/welding_defect_model6/weights/best.onnx"

# If you have the .pt file, use it instead:
# MODEL_PATH = "runs/detect/welding_defect_model6/weights/best.pt"

model = YOLO(MODEL_PATH)

results = model("dataset/valid/images", save=True)

print("Detection completed!")
