"""Flask web app: upload image, pick category, EfficientAD detects defect."""

import base64
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from torchvision.transforms.v2 import Resize

from anomalib.models import EfficientAd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (256, 256)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

REGISTRY_PATH = Path("models/registry.json")
METRICS_ROOT = Path("metrics")


# ============================================================
# REGISTRY
# ============================================================


def load_registry():
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"Model registry not found: {REGISTRY_PATH}")

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    for category, meta in registry.items():
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.exists():
            raise RuntimeError(
                f"Checkpoint for '{category}' missing: {checkpoint}"
            )

        if meta["threshold"] is None:
            raise RuntimeError(
                f"Threshold not set for '{category}'. "
                "Run src/calibrate_thresholds.py first."
            )

    return registry


REGISTRY = load_registry()


def load_category_metrics(category: str) -> dict:
    path = METRICS_ROOT / f"{category}.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# DEVICE + LAZY MODEL CACHE
# ============================================================


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
MODEL_CACHE = {}
CACHE_LOCK = threading.Lock()


def get_model(category: str):
    """Lazy-load a category model. Thread-safe."""
    with CACHE_LOCK:
        if category in MODEL_CACHE:
            return MODEL_CACHE[category]

        if category not in REGISTRY:
            raise ValueError(f"Unknown category: {category}")

        checkpoint = REGISTRY[category]["checkpoint"]
        print(f"Loading model: {category} -> {checkpoint}")
        print(f"Using device: {DEVICE}")

        model = EfficientAd.load_from_checkpoint(
            checkpoint,
            map_location=DEVICE,
        )
        model.to(DEVICE)
        model.eval()
        resize = Resize(IMAGE_SIZE, antialias=True)

        MODEL_CACHE[category] = (model, resize)
        return model, resize


# ============================================================
# INFERENCE
# ============================================================


@torch.inference_mode()
def predict(category: str, image_bgr):
    """Run EfficientAD on a BGR image. Return score and anomaly map."""
    model, resize = get_model(category)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = resize(tensor).unsqueeze(0).to(DEVICE)

    prediction = model.post_processor(model.model(tensor))

    score = float(prediction.pred_score[0].detach().cpu())
    anomaly_map = prediction.anomaly_map[0].detach().cpu().squeeze().numpy()

    return score, anomaly_map


def make_heatmap_overlay(image_bgr, anomaly_map, alpha=0.4):
    """Blend JET heatmap over the original image. Return BGR image."""
    clipped = np.clip(anomaly_map, 0.0, 1.0)
    heatmap = (clipped * 255).astype(np.uint8)
    heatmap = cv2.resize(
        heatmap,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)


def encode_png(image_bgr):
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/categories")
def categories():
    payload = []
    for category, meta in REGISTRY.items():
        metrics = load_category_metrics(category)
        payload.append(
            {
                "name": category,
                "tier": meta["tier"],
                "threshold": meta["threshold"],
                "metrics": metrics,
            }
        )

    return jsonify({"categories": payload})


@app.post("/predict")
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    category = request.form.get("category", "").strip().lower()
    if not category:
        return jsonify({"error": "No category selected."}), 400

    if category not in REGISTRY:
        return jsonify({"error": f"Unknown or unsupported category: {category}."}), 400

    data = file.read()
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image."}), 400

    threshold = request.form.get("threshold", type=float)
    if threshold is None:
        threshold = REGISTRY[category]["threshold"]

    try:
        score, anomaly_map = predict(category, image)
    except Exception as error:
        return jsonify({"error": f"Inference failed: {error}"}), 500

    is_defect = score >= threshold
    overlay = make_heatmap_overlay(image, anomaly_map)

    return jsonify(
        {
            "category": category,
            "tier": REGISTRY[category]["tier"],
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "is_defect": bool(is_defect),
            "result": "DEFECT" if is_defect else "NORMAL",
            "heatmap": encode_png(overlay),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
