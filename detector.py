"""
Lightweight YOLOv8 ONNX inference module.
No PyTorch / Ultralytics needed — only ONNX Runtime + OpenCV.
"""

import time
import numpy as np
import cv2
from pathlib import Path

CLASS_NAMES = {0: "crack", 1: "porosity", 2: "undercut"}
INPUT_SIZE = 416  # Model was trained at this resolution


class YOLODetector:
    """ONNX-based YOLOv8 detector — memory efficient (~200MB)."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check if external data file exists alongside the model
        data_file = Path(str(model_path) + ".data")
        if data_file.exists():
            print(f"  [ONNX Detector] External data file found: {data_file.name}")

        # Set session options for better compatibility
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Get model input shape for validation
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        print(f"  [ONNX Detector] Input: {self.input_name} {input_shape}")
        print(f"  [ONNX Detector] Output: {output_shape}")

        self._loaded = True
        print(f"  [ONNX Detector] Model loaded from {model_file.name}")

    def preprocess(self, img_rgb: np.ndarray):
        """Resize and normalize image to model input size."""
        h, w = img_rgb.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw = int(w * scale)
        nh = int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh))

        # Letterbox padding to square
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        dx = (INPUT_SIZE - nw) // 2
        dy = (INPUT_SIZE - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        # Convert to NCHW [1, 3, 416, 416], float32, normalized to [0, 1]
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, dx, dy

    def postprocess(
        self,
        output: np.ndarray,
        conf_threshold: float,
        iou_threshold: float = 0.45,
    ):
        """
        Decode YOLOv8 ONNX output into detections.
        Output shape: [1, 7, 3549] where:
          - rows 0-3: box coordinates (cx, cy, w, h) or (x1, y1, x2, y2)
          - rows 4-6: class logits (raw, need sigmoid)
        """
        output = output[0]  # [7, 3549]

        # Apply sigmoid to class scores
        scores = 1.0 / (1.0 + np.exp(-output[4:, :]))  # [3, 3549]
        max_scores = scores.max(axis=0)  # [3549]
        class_ids = scores.argmax(axis=0)  # [3549]

        # Filter by confidence
        mask = max_scores >= conf_threshold
        if not mask.any():
            return []

        # Boxes — check format by analyzing values
        # YOLOv8 ONNX export can be xywh or xyxy depending on version
        row0 = output[0, mask]
        row1 = output[1, mask]
        row2 = output[2, mask]
        row3 = output[3, mask]

        # Heuristic: if row2/row3 values are generally larger than row0/row1,
        # it's xyxy format. If row2/row3 are smaller (width/height), it's xywh.
        if len(row0) > 0:
            avg_02_diff = np.mean(row2 - row0)
            if avg_02_diff > 0 and np.mean(row2) > np.mean(row0):
                # xyxy format
                x1 = row0
                y1 = row1
                x2 = row2
                y2 = row3
            else:
                # xywh (center format) — convert to xyxy
                cx, cy, w, h = row0, row1, row2, row3
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
        else:
            return []

        sc = max_scores[mask]
        cl = class_ids[mask]

        # Filter out invalid boxes (negative area)
        valid = (x2 > x1) & (y2 > y1)
        if not valid.any():
            return []

        x1 = x1[valid]
        y1 = y1[valid]
        x2 = x2[valid]
        y2 = y2[valid]
        sc = sc[valid]
        cl = cl[valid]

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        keep = self._nms(boxes, sc, iou_threshold)

        detections = []
        for i in keep:
            detections.append({
                "class_id": int(cl[i]),
                "class_name": CLASS_NAMES.get(int(cl[i]), "unknown"),
                "confidence": round(float(sc[i]), 4),
                "bbox": [round(float(x1[i]), 2),
                         round(float(y1[i]), 2),
                         round(float(x2[i]), 2),
                         round(float(y2[i]), 2)],
            })

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
        """Simple NMS implementation (no torch dependency)."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            # IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0)
            order = order[1:][iou <= iou_thresh]

        return keep

    def detect(self, img_rgb: np.ndarray, conf_threshold: float = 0.25):
        """
        Full detection pipeline.
        Returns list of dicts with class_name, confidence, bbox (xyxy in original image coords).
        """
        orig_h, orig_w = img_rgb.shape[:2]

        # Preprocess
        blob, scale, dx, dy = self.preprocess(img_rgb)

        # Inference
        t0 = time.time()
        output = self.session.run(None, {self.input_name: blob})[0]
        infer_time = time.time() - t0

        # Postprocess (boxes are in 416x416 letterbox space)
        detections = self.postprocess(output, conf_threshold)

        # Scale boxes from 416x416 letterbox to original image
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Remove letterbox padding
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale
            # Clamp to image boundaries
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            det["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

        return detections, infer_time
