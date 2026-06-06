from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  # lightweight model

    model.train(
        data="data.yaml",
        epochs=15,              # fast training
        imgsz=416,              # smaller image size
        batch=4,                # for low RAM
        name="welding_defect_model"  # fixed folder name
    )

if __name__ == "__main__":
    train()