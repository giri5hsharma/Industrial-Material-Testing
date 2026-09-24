"""Calibrate a detection threshold per class from MVTec AD test data.

Loads each checkpoint kept in models/registry.json, scores every
image in the test split, and writes a threshold tuned to separate
good samples from defects.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from anomalib.models import EfficientAd


DATASET_ROOT = Path("datasets/MVTecAD")
REGISTRY_PATH = Path("models/registry.json")
IMAGE_SIZE = (256, 256)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def score_image(model, resize, device, image_path: Path) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = resize(tensor).unsqueeze(0).to(device)

    prediction = model.post_processor(model.model(tensor))
    return float(prediction.pred_score[0].detach().cpu())


def calibrate_category(category: str, checkpoint_path: Path, device: torch.device):
    print(f"\nCalibrating {category}...")

    model = EfficientAd.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )
    model.to(device)
    model.eval()
    resize = Resize(IMAGE_SIZE, antialias=True)

    test_root = DATASET_ROOT / category / "test"

    good_scores = []
    defect_scores = []

    for subdir in sorted(test_root.iterdir()):
        if not subdir.is_dir():
            continue

        for image_path in sorted(subdir.glob("*.png")):
            score = score_image(model, resize, device, image_path)

            if subdir.name == "good":
                good_scores.append(score)
            else:
                defect_scores.append(score)

    del model
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    good_scores = np.array(good_scores, dtype=np.float32)
    defect_scores = np.array(defect_scores, dtype=np.float32)

    good_max = float(good_scores.max())
    good_median = float(np.median(good_scores))
    defect_min = float(defect_scores.min())
    defect_median = float(np.median(defect_scores))

    # Default formula: place threshold between good max and defect median.
    if defect_median > good_max:
        threshold = good_max + 0.4 * (defect_median - good_max)
    else:
        # Distributions overlap: use a small margin above good max.
        threshold = good_max + 0.01

    print(f"  good   n={len(good_scores):3d}  max={good_max:.4f}  median={good_median:.4f}")
    print(f"  defect n={len(defect_scores):3d}  min={defect_min:.4f}  median={defect_median:.4f}")
    print(f"  threshold={threshold:.4f}")

    return threshold


def main():
    device = get_device()
    print(f"Using device: {device}")

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    for category, meta in registry.items():
        checkpoint = Path(meta["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")

        threshold = calibrate_category(category, checkpoint, device)
        meta["threshold"] = round(threshold, 4)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)

    print(f"\nUpdated thresholds written to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
