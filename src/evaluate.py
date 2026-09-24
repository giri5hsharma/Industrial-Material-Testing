import argparse
import json
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("datasets/MVTecAD")

MODEL_ROOT = Path("models")

METRICS_ROOT = Path("metrics")

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "leather",
    "metal_nut",
    "screw",
    "tile",
    "transistor",
    "wood",
]


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_metrics(test_results) -> dict:

    if not test_results:
        return {}

    metrics = test_results[0]

    metric_names = [
        "image_AUROC",
        "image_F1Score",
        "pixel_AUROC",
        "pixel_F1Score",
    ]

    clean_metrics = {}

    for name in metric_names:

        if name not in metrics:
            continue

        value = metrics[name]

        # Convert torch tensor -> Python float
        if hasattr(value, "item"):
            value = value.item()

        clean_metrics[name] = float(value)

    return clean_metrics


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    category: str,
    metrics: dict,
):

    METRICS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        METRICS_ROOT
        / f"{category}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Metrics saved to: {output_path}"
    )


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    category: str,
    metrics: dict,
):

    print()
    print("=" * 70)
    print(
        f"EVALUATION RESULTS: {category}"
    )
    print("=" * 70)
    print()

    if not metrics:

        print(
            "No metrics were returned."
        )

        print()

        return

    for name, value in metrics.items():

        print(
            f"{name:<20}: {value:.6f}"
        )

    print()


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    category: str,
):

    checkpoint_path = (
        MODEL_ROOT
        / category
        / "model.ckpt"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"EVALUATING MODEL: {category}"
    )
    print("=" * 70)
    print()

    print(
        f"Checkpoint:"
    )

    print(
        f"  {checkpoint_path}"
    )

    print()

    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category=category,

        train_batch_size=1,
        eval_batch_size=1,

        num_workers=0,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = EfficientAd.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
    )

    # --------------------------------------------------------
    # Evaluation engine
    # --------------------------------------------------------

    engine = Engine(
        accelerator=device,
        devices=1,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print(
        "Running evaluation..."
    )

    print()

    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,
    )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    metrics = extract_metrics(
        test_results
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_metrics(
        category,
        metrics,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_metrics(
        category,
        metrics,
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained EfficientAD "
            "model on MVTec AD."
        )
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=CATEGORIES,
        help=(
            "MVTec category to evaluate."
        ),
    )

    args = parser.parse_args()

    evaluate_model(
        args.category
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()