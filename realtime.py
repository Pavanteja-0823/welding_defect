from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        frame = r.plot()

    cv2.imshow("Welding Defect Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()