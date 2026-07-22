"""
Train YOLOv8 model for welding defect detection.
Classes: crack, porosity, undercut
"""

from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.pt")  # lightweight model

    model.train(
        data="data.yaml",
        epochs=15,              # fast training
        imgsz=416,              # smaller image size (matches detector.py)
        batch=4,                # for low RAM
        name="welding_defect_model"  # fixed folder name
    )

    # Export to ONNX for lightweight production deployment
    print("\nExporting to ONNX...")
    best_model = YOLO("runs/detect/welding_defect_model/weights/best.pt")
    best_model.export(format="onnx", imgsz=416, simplify=True)
    print("ONNX export complete!")


if __name__ == "__main__":
    train()
