import base64
import io
import time
import argparse
import threading
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("runs/detect/welding_defect_model6/weights/best.pt")
CLASS_NAMES = {0: "crack", 1: "porosity", 2: "undercut"}

# Thread-safe model loading
_model = None
_model_lock = threading.Lock()
_model_loading = False
_model_ready = False
_model_error = None


def _load_model():
    """Internal: actually load the model."""
    global _model, _model_ready, _model_error
    try:
        print(f"  [Background] Loading model from {MODEL_PATH}...")
        _model = YOLO(str(MODEL_PATH))
        _model_ready = True
        print(f"  [Background] Model loaded successfully!")
    except Exception as e:
        _model_error = str(e)
        print(f"  [Background] Model loading FAILED: {e}")


def get_model():
    """Get the model — triggers background load on first call if not started."""
    global _model_loading
    if _model is None and not _model_loading:
        with _model_lock:
            if _model is None and not _model_loading:
                _model_loading = True
                t = threading.Thread(target=_load_model, daemon=True)
                t.start()
    return _model


@app.route("/api/status")
def status():
    """Return model loading status so the frontend can show progress."""
    return jsonify({
        "model_ready": _model_ready,
        "model_loading": _model_loading,
        "model_error": _model_error,
    })


def classify_weld(detections):
    if not detections:
        return {
            "status": "GOOD",
            "label": "Good Weld",
            "description": "No defects detected. The weld appears to be in good condition.",
            "severity": "none",
            "color": "#22c55e",
        }
    defect_types = set()
    max_conf = 0
    for det in detections:
        defect_types.add(det["class_name"])
        max_conf = max(max_conf, det["confidence"])
    severity = "low"
    if max_conf > 0.7:
        severity = "high"
    elif max_conf > 0.4:
        severity = "medium"
    defects_str = ", ".join(sorted(defect_types))
    return {
        "status": "BAD",
        "label": f"Bad Weld — {defects_str}",
        "description": f"Defects detected: {defects_str}. Confidence: {max_conf:.1%}.",
        "severity": severity,
        "color": "#ef4444",
    }


def get_defect_description(class_name):
    descriptions = {
        "crack": "Crack — linear fracture in the weld metal or heat-affected zone.",
        "porosity": "Porosity — gas pockets or voids trapped in the solidified weld.",
        "undercut": "Undercut — groove melted into the base metal adjacent to the weld toe.",
    }
    return descriptions.get(class_name, class_name)


def draw_detections(img_bgr, detections):
    """Draw bounding boxes and labels for filtered detections on the image."""
    img_copy = img_bgr.copy()
    colors = {
        "crack": (0, 0, 230),
        "porosity": (0, 165, 255),
        "undercut": (255, 100, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        cls_name = det["class_name"]
        color = colors.get(cls_name, (0, 200, 0))

        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Draw filled label background
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)

        # Draw label text
        cv2.putText(img_copy, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_copy


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Parse optional confidence threshold
    try:
        threshold_str = request.form.get("confidence_threshold", "0.25")
        confidence_threshold = float(threshold_str)
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
    except (ValueError, TypeError):
        confidence_threshold = 0.25

    try:
        t_start = time.time()

        image_bytes = file.read()
        if len(image_bytes) > 20 * 1024 * 1024:
            return jsonify({"error": "File too large. Maximum 20MB."}), 400

        image = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        model = get_model()
        if model is None:
            return jsonify({"error": "Model is still loading. Please wait a moment and try again.", "model_loading": True}), 503

        results = model(img_np)
        t_inference = time.time()

        # Collect all raw detections from model
        all_detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    all_detections.append({
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES.get(cls_id, "unknown"),
                        "confidence": round(confidence, 4),
                        "bbox": [round(v, 2) for v in xyxy],
                    })

        # Filter by confidence threshold
        detections = [d for d in all_detections if d["confidence"] >= confidence_threshold]

        classification = classify_weld(detections)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        _, orig_buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

        # Manually draw only filtered detections
        annotated_bgr = draw_detections(img_bgr, detections)
        _, anno_buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(anno_buffer).decode("utf-8")

        defect_details = [
            {"class_name": d["class_name"], "confidence": d["confidence"],
             "description": get_defect_description(d["class_name"])}
            for d in detections
        ]

        t_end = time.time()

        return jsonify({
            "success": True,
            "classification": classification,
            "detections": detections,
            "defect_details": defect_details,
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "num_defects": len(detections),
            "total_raw_detections": len(all_detections),
            "confidence_threshold": confidence_threshold,
            "processing_time_ms": round((t_end - t_start) * 1000),
            "inference_time_ms": round((t_end - t_inference) * 1000),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🔬 Welding Defect Detection UI")
    print(f"  🌐 http://{args.host}:{args.port}")
    print(f"  ⏳ Model loading in background... page loads instantly!")
    print(f"{'='*60}\n")

    # Start model loading immediately
    get_model()

    app.run(host=args.host, port=args.port, debug=True)
