---
title: Welding Defect Detection
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Welding Defect Detection — YOLOv8

Detect welding defects (cracks, porosity, undercut) using a trained YOLOv8 model.

## 🔬 How it works

Upload a weld image, and the AI model analyzes it for three types of defects:

| Defect | Description |
|--------|-------------|
| 💥 **Crack** | Linear fracture in the weld metal or heat-affected zone |
| 🕳️ **Porosity** | Gas pockets or voids trapped in the solidified weld |
| 🔻 **Undercut** | Groove melted into the base metal adjacent to the weld toe |

## 🚀 Local Development

pip install -r requirements.txt
python app.py

Open http://localhost:5000 in your browser.

## 🧠 Training

python train.py

## 📦 Deploy on Hugging Face Spaces (Free, 16GB RAM)

1. Go to https://huggingface.co/new-space
2. Name: welding-defect-detection
3. SDK: Docker
4. Space Hardware: Free (2 vCPU · 16GB RAM)
5. Connect your GitHub repo or push directly

Built with YOLOv8 · Flask · PyTorch