import argparse
import gc
import json
import shutil
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


# ============================================================
# CONFIGURATION
# ============================================================

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

DATASET_ROOT = Path("datasets/MVTecAD")

MODEL_ROOT = Path("models")

METRICS_ROOT = Path("metrics")

DEFAULT_EPOCHS = 20


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
# CHECKPOINT
# ============================================================

def get_best_checkpoint(
    engine: Engine,
) -> Path | None:
    """
    Get the actual best checkpoint path produced
    by Lightning/Anomalib.
    """

    best_path = engine.best_model_path

    if best_path:
        path = Path(best_path)

        if path.exists():
            return path

    return None


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_metrics(
    test_results,
) -> dict:

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

        if hasattr(value, "item"):
            value = value.item()

        clean_metrics[name] = float(value)

    return clean_metrics


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
            "No evaluation metrics returned."
        )

        print()

        return

    for name, value in metrics.items():

        print(
            f"{name:<20}: {value:.6f}"
        )

    print()


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
# TRAIN + EVALUATE ONE CATEGORY
# ============================================================

def train_category(
    category: str,
    epochs: int,
    device: str,
) -> bool:

    print()
    print("=" * 70)
    print(
        f"TRAINING CATEGORY: {category}"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    output_dir = (
        MODEL_ROOT
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_checkpoint = (
        output_dir
        / "model.ckpt"
    )

    # --------------------------------------------------------
    # Skip existing checkpoint
    # --------------------------------------------------------

    if final_checkpoint.exists():

        print(
            f"Checkpoint already exists:"
        )

        print(
            f"  {final_checkpoint}"
        )

        print(
            "Skipping training."
        )

        return True

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        f"Preparing MVTec AD category: {category}"
    )

    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category=category,

        # EfficientAD requires batch size = 1
        train_batch_size=1,

        eval_batch_size=1,

        # Safer on macOS
        num_workers=0,
    )

    datamodule.prepare_data()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = EfficientAd(
        model_size="small",
        lr=1e-4,
    )

    # --------------------------------------------------------
    # Engine
    # --------------------------------------------------------

    engine = Engine(
        max_epochs=epochs,
        accelerator=device,
        devices=1,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        f"Starting training for: {category}"
    )

    print(
        f"Epochs: {epochs}"
    )

    print(
        f"Device: {device}"
    )

    print()

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    # --------------------------------------------------------
    # FIND BEST CHECKPOINT
    # --------------------------------------------------------

    best_checkpoint = get_best_checkpoint(
        engine
    )

    print()

    if best_checkpoint is None:

        print(
            "WARNING: No best checkpoint was found."
        )

        print(
            "The trained model will be evaluated "
            "using its current in-memory weights."
        )

    else:

        print(
            f"Best checkpoint:"
        )

        print(
            f"  {best_checkpoint}"
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"EVALUATING: {category}"
    )
    print("=" * 70)
    print()

    if best_checkpoint is not None:

        test_results = engine.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(
                best_checkpoint
            ),
        )

    else:

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
    # Print metrics
    # --------------------------------------------------------

    print_metrics(
        category,
        metrics,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        category,
        metrics,
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    if best_checkpoint is None:

        print(
            "ERROR: Cannot save model because "
            "no checkpoint was produced."
        )

        return False

    shutil.copy2(
        best_checkpoint,
        final_checkpoint,
    )

    print()
    print("=" * 70)
    print(
        f"MODEL SAVED: {category}"
    )
    print("=" * 70)
    print()

    print(
        f"Checkpoint:"
    )

    print(
        f"  {final_checkpoint}"
    )

    print()

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    del datamodule
    del engine

    gc.collect()

    if torch.backends.mps.is_available():

        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate EfficientAD "
            "models on MVTec AD."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Number of epochs per category."
        ),
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=CATEGORIES,
        help=(
            "Train only one category."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain even if a checkpoint "
            "already exists."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    if args.category is not None:

        categories = [
            args.category
        ]

    else:

        categories = CATEGORIES

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
        "MVTec AD - EfficientAD "
        "Multi-Category Training"
    )
    print("=" * 70)
    print()

    print(
        f"Device:              {device}"
    )

    print(
        f"Epochs/category:     {args.epochs}"
    )

    print(
        f"Dataset root:        {DATASET_ROOT}"
    )

    print(
        f"Model root:          {MODEL_ROOT}"
    )

    print(
        f"Metrics root:        {METRICS_ROOT}"
    )

    print()

    print(
        "Categories:"
    )

    for category in categories:

        print(
            f"  - {category}"
        )

    print()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    successful = []
    failed = []

    for category in categories:

        model_path = (
            MODEL_ROOT
            / category
            / "model.ckpt"
        )

        # ----------------------------------------------------
        # Force retraining
        # ----------------------------------------------------

        if (
            args.force
            and model_path.exists()
        ):

            print(
                f"Removing existing model:"
            )

            print(
                f"  {model_path}"
            )

            model_path.unlink()

        try:

            success = train_category(
                category=category,
                epochs=args.epochs,
                device=device,
            )

            if success:

                successful.append(
                    category
                )

            else:

                failed.append(
                    category
                )

        except KeyboardInterrupt:

            print()
            print(
                "Training interrupted by user."
            )

            break

        except Exception as error:

            print()
            print("=" * 70)
            print(
                f"FAILED: {category}"
            )
            print("=" * 70)
            print()

            print(
                f"Error: {error}"
            )

            failed.append(
                category
            )

            # Cleanup
            gc.collect()

            if torch.backends.mps.is_available():

                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINING + EVALUATION COMPLETE"
    )
    print("=" * 70)
    print()

    print("Successful:")

    if successful:

        for category in successful:

            print(
                f"  ✓ {category}"
            )

    else:

        print("  None")

    print()

    print("Failed:")

    if failed:

        for category in failed:

            print(
                f"  ✗ {category}"
            )

    else:

        print("  None")

    print()

    print(
        f"Models : {MODEL_ROOT.resolve()}"
    )

    print(
        f"Metrics: {METRICS_ROOT.resolve()}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()