# 🔬 Weld Inspector — AI Welding Defect Detection

Real-time welding defect detection powered by YOLOv8 and ONNX Runtime. Upload weld images and instantly get AI-powered quality assessment with bounding box annotations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-green?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

- **3 Defect Classes**: Crack, Porosity, Undercut
- **Real-time Detection**: ONNX Runtime inference (~100ms per image)
- **Beautiful UI**: Dark-themed, responsive interface with drag-and-drop upload
- **Confidence Slider**: Adjust detection sensitivity in real-time
- **Side-by-Side Comparison**: Original vs annotated image view
- **Classification Report**: Good/Bad weld verdict with severity levels
- **Analysis History**: Track past detections in-session
- **Batch API**: Analyze multiple images in one request
- **Lightweight**: No PyTorch needed in production (~200MB total)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask + Gunicorn |
| AI Model | YOLOv8n (ONNX export, 416×416) |
| Inference | ONNX Runtime (CPU) |
| Image Processing | OpenCV + Pillow |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker / Render / Hugging Face Spaces |

## 📁 Project Structure

```
welding_defect/
├── app.py                 # Flask server (main entry point)
├── detector.py            # ONNX inference engine
├── train.py               # Model training script
├── detect.py              # Batch detection on validation set
├── realtime.py            # Webcam real-time detection
├── data.yaml              # Training data config
├── requirements.txt       # Production dependencies
├── Dockerfile             # Docker container config
├── Procfile               # Render/Heroku deployment
├── templates/
│   └── index.html         # Web UI (single-page app)
├── dataset/
│   ├── train/images/      # Training images
│   ├── valid/images/      # Validation images
│   └── test/images/       # Test images
├── runs/detect/
│   └── welding_defect_model6/
│       └── weights/
│           ├── best.onnx      # Production model
│           └── best.onnx.data # Model weights (external)
└── yolov8n.pt             # Base model for training
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/welding-defect-detection.git
cd welding-defect-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python app.py --port 5001
```

Open **http://127.0.0.1:5001** in your browser.

The AI model loads in the background (~10-30 seconds on first start). The header badge turns green when ready.

### Run with Demo Image

A sample weld image is included at `dataset/sample/demo_weld.jpg` for quick testing.

## 📡 API Endpoints

### `GET /` — Web Interface
Returns the detection UI.

### `GET /api/status` — Model Status
```json
{"model_ready": true, "model_loading": false, "model_error": null}
```

### `GET /api/health` — Health Check
```json
{"status": "healthy", "model_ready": true, "uptime_seconds": 120.5}
```

### `POST /api/detect` — Single Image Detection
```bash
curl -X POST http://localhost:5001/api/detect \
  -F "image=@weld_image.jpg" \
  -F "confidence_threshold=0.25"
```

**Response:**
```json
{
  "success": true,
  "classification": {
    "status": "BAD",
    "label": "Bad Weld — crack, porosity",
    "severity": "high",
    "color": "#ef4444"
  },
  "detections": [
    {"class_id": 0, "class_name": "crack", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
  ],
  "num_defects": 2,
  "processing_time_ms": 145,
  "annotated_image": "<base64>",
  "original_image": "<base64>"
}
```

### `POST /api/batch` — Batch Detection (Multiple Images)
```bash
curl -X POST http://localhost:5001/api/batch \
  -F "images=@weld1.jpg" \
  -F "images=@weld2.jpg" \
  -F "confidence_threshold=0.3"
```

**Response:**
```json
{
  "success": true,
  "results": [...],
  "total_images": 2,
  "total_defects": 5,
  "summary": {"good": 0, "bad": 2}
}
```

## 🏋️ Training

To retrain the model on your own dataset:

```bash
# Ensure dataset is organized:
# dataset/train/images/ + dataset/train/labels/
# dataset/valid/images/ + dataset/valid/labels/

python train.py
```

This trains YOLOv8n for 15 epochs at 416×416 resolution and auto-exports to ONNX.

## 🐳 Docker Deployment

```bash
# Build
docker build -t weld-inspector .

# Run
docker run -p 7860:7860 -e PORT=7860 weld-inspector
```

## ☁️ Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
5. Deploy!

## 🤗 Deploy to Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Push this repo to the Space
3. The Dockerfile handles everything automatically

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 38.2% |
| mAP@50-95 | 21.4% |
| Precision | 38.6% |
| Recall | 51.6% |
| Epochs | 15 |
| Image Size | 416×416 |
| Inference | ~100ms (CPU) |

> **Note**: Performance can be improved with more training data, larger models (yolov8s/m), more epochs, and data augmentation.

## 📝 License

MIT License — free for personal and commercial use.

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Flask](https://flask.palletsprojects.com/)
