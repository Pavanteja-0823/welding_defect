from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")  # change if needed

results = model("dataset/valid/images", save=True)

print("Detection completed!")