"""
Real-time welding defect detection using webcam.
Uses the latest trained model (welding_defect_model6).

Note: Requires ultralytics and a .pt model for real-time inference.
      ONNX model can also be used but .pt is more common for this use case.
"""

from ultralytics import YOLO
import cv2

# Use the latest trained model (ONNX works here too)
MODEL_PATH = "runs/detect/welding_defect_model6/weights/best.onnx"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("Starting real-time welding defect detection...")
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame)

    for r in results:
        frame = r.plot()

    cv2.imshow("Welding Defect Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
