import base64
import io
import os
import time
import argparse
import threading
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

from detector import YOLODetector

# ——— App Setup ———
app = Flask(__name__)
CORS(app)

# Request logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_BATCH_SIZE = 10  # Max images in a batch request
MODEL_PATH_STR = "runs/detect/welding_defect_model6/weights/best.onnx"
CLASS_NAMES = {0: "crack", 1: "porosity", 2: "undercut"}

# Track uptime
_start_time = time.time()

# ——— Thread-safe Model Loading ———
_model = None
_model_lock = threading.Lock()
_model_loading = False
_model_ready = False
_model_error = None


def _load_model():
    """Internal: actually load the ONNX model (lightweight, no PyTorch)."""
    global _model, _model_ready, _model_error, _model_loading
    try:
        logger.info(f"Loading ONNX model from {MODEL_PATH_STR}...")

        if not os.path.exists(MODEL_PATH_STR):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH_STR}. "
                f"Working directory: {os.getcwd()}"
            )

        _model = YOLODetector(MODEL_PATH_STR)
        _model_ready = True
        _model_loading = False
        logger.info("ONNX model loaded successfully!")
    except Exception as e:
        _model_error = str(e)
        _model_loading = False
        logger.error(f"Model loading FAILED: {e}")


def get_model():
    """Get the model — triggers background load on first call if not started."""
    global _model_loading
    if _model is None and not _model_loading and _model_error is None:
        with _model_lock:
            if _model is None and not _model_loading and _model_error is None:
                _model_loading = True
                t = threading.Thread(target=_load_model, daemon=True)
                t.start()
    return _model


# ——— Helper Functions ———

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

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_copy, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_copy


def process_image(image_bytes, confidence_threshold=0.25):
    """Process a single image and return detection results."""
    image = Image.open(io.BytesIO(image_bytes))
    img_np = np.array(image)

    # Handle various image formats
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.ndim == 3 and img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif img_np.ndim == 3 and img_np.shape[-1] != 3:
        raise ValueError(f"Unsupported image format: {img_np.shape}")

    model = get_model()
    if model is None:
        if _model_error:
            raise RuntimeError(f"Model failed to load: {_model_error}")
        raise RuntimeError("Model is still loading")

    # Run inference
    detections, infer_time = model.detect(img_np, conf_threshold=confidence_threshold)
    classification = classify_weld(detections)

    # Encode images
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, orig_buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_b64 = base64.b64encode(orig_buffer).decode("utf-8")

    annotated_bgr = draw_detections(img_bgr, detections)
    _, anno_buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(anno_buffer).decode("utf-8")

    defect_details = [
        {"class_name": d["class_name"], "confidence": d["confidence"],
         "description": get_defect_description(d["class_name"])}
        for d in detections
    ]

    return {
        "classification": classification,
        "detections": detections,
        "defect_details": defect_details,
        "original_image": original_b64,
        "annotated_image": annotated_b64,
        "num_defects": len(detections),
        "total_raw_detections": len(detections),
        "confidence_threshold": confidence_threshold,
        "inference_time_ms": round(infer_time * 1000),
    }


# ——— Request Logging Middleware ———

@app.before_request
def log_request():
    """Log incoming requests."""
    if request.path.startswith('/api/'):
        logger.info(f"{request.method} {request.path} from {request.remote_addr}")


@app.after_request
def log_response(response):
    """Log response status."""
    if request.path.startswith('/api/') and response.status_code >= 400:
        logger.warning(f"{request.method} {request.path} → {response.status_code}")
    return response


# ——— Routes ———

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/health")
def health():
    """Health check endpoint for monitoring and load balancers."""
    return jsonify({
        "status": "healthy",
        "model_ready": _model_ready,
        "model_loading": _model_loading,
        "model_error": _model_error,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "version": "1.0.0",
    })


@app.route("/api/status")
def status():
    """Return model loading status for the frontend."""
    return jsonify({
        "model_ready": _model_ready,
        "model_loading": _model_loading,
        "model_error": _model_error,
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    """Detect welding defects in a single uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Parse confidence threshold
    try:
        threshold_str = request.form.get("confidence_threshold", "0.25")
        confidence_threshold = float(threshold_str)
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
    except (ValueError, TypeError):
        confidence_threshold = 0.25

    try:
        t_start = time.time()

        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Maximum {MAX_FILE_SIZE // (1024*1024)}MB."}), 400

        result = process_image(image_bytes, confidence_threshold)

        t_end = time.time()
        result["success"] = True
        result["processing_time_ms"] = round((t_end - t_start) * 1000)

        logger.info(f"Detection complete: {result['num_defects']} defects in {result['processing_time_ms']}ms")
        return jsonify(result)

    except RuntimeError as e:
        if "still loading" in str(e):
            return jsonify({"error": str(e), "model_loading": True}), 503
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_detect():
    """Detect welding defects in multiple images at once."""
    files = request.files.getlist("images")

    if not files or len(files) == 0:
        return jsonify({"error": "No images provided. Use form field 'images'."}), 400

    if len(files) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Too many images. Maximum {MAX_BATCH_SIZE} per batch."}), 400

    # Parse confidence threshold
    try:
        threshold_str = request.form.get("confidence_threshold", "0.25")
        confidence_threshold = float(threshold_str)
        confidence_threshold = max(0.0, min(1.0, confidence_threshold))
    except (ValueError, TypeError):
        confidence_threshold = 0.25

    t_start = time.time()
    results = []
    total_defects = 0
    summary = {"good": 0, "bad": 0}

    for i, file in enumerate(files):
        if file.filename == "":
            continue

        try:
            image_bytes = file.read()
            if len(image_bytes) > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File too large",
                })
                continue

            result = process_image(image_bytes, confidence_threshold)
            result["success"] = True
            result["filename"] = file.filename

            # Don't include base64 images in batch (too large) — just detections
            del result["original_image"]
            del result["annotated_image"]

            total_defects += result["num_defects"]
            if result["classification"]["status"] == "GOOD":
                summary["good"] += 1
            else:
                summary["bad"] += 1

            results.append(result)

        except RuntimeError as e:
            if "still loading" in str(e):
                return jsonify({"error": "Model is still loading. Please wait.", "model_loading": True}), 503
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })

    t_end = time.time()

    logger.info(f"Batch detection: {len(results)} images, {total_defects} defects in {round((t_end - t_start) * 1000)}ms")

    return jsonify({
        "success": True,
        "results": results,
        "total_images": len(results),
        "total_defects": total_defects,
        "summary": summary,
        "processing_time_ms": round((t_end - t_start) * 1000),
        "confidence_threshold": confidence_threshold,
    })


# ——— Upload Size Validation ———

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum 20MB."}), 413


# Set max content length
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE * MAX_BATCH_SIZE  # Allow batch uploads


# ——— Startup ———

# Start model loading in background when module is imported
get_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weld Inspector — AI Defect Detection Server")
    parser.add_argument("--port", type=int, default=None, help="Port (default: PORT env or 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 5000))
    debug = args.debug or os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")

    print("=" * 60)
    print("  🔬 Weld Inspector — AI Defect Detection")
    print("=" * 60)
    print(f"  Server:  http://{args.host}:{port}")
    print(f"  Model:   {MODEL_PATH_STR}")
    print(f"  Exists:  {os.path.exists(MODEL_PATH_STR)}")
    print(f"  Classes: {', '.join(CLASS_NAMES.values())}")
    print("  Status:  Model loading in background...")
    print("=" * 60)
    print("  Endpoints:")
    print(f"    GET  /             — Web UI")
    print(f"    GET  /api/status   — Model status")
    print(f"    GET  /api/health   — Health check")
    print(f"    POST /api/detect   — Single image detection")
    print(f"    POST /api/batch    — Batch detection (up to {MAX_BATCH_SIZE} images)")
    print("=" * 60)

    app.run(host=args.host, port=port, debug=debug)
