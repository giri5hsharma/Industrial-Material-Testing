import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "metrics"
OUTPUT_PATH = METRICS_DIR / "efficientad_performance_metrics.png"
EXPECTED_CATEGORIES = ("bottle", "cable", "leather", "metal_nut", "tile", "wood")
METRIC_KEYS = (
    ("image_AUROC", "Image-level AUROC", "#176b87"),
    ("image_F1Score", "Image-level F1-score", "#c94c4c"),
    ("pixel_AUROC", "Pixel-level AUROC", "#d69e2e"),
    ("pixel_F1Score", "Pixel-level F1-score", "#6b4f9b"),
)


def main() -> None:
    metrics_by_category = {}
    for category in EXPECTED_CATEGORIES:
        metric_file = METRICS_DIR / f"{category}.json"
        if not metric_file.exists():
            raise FileNotFoundError(f"Missing metric file: {metric_file}")

        with metric_file.open(encoding="utf-8") as file:
            metrics_by_category[category] = json.load(file)

    categories = [category.replace("_", " ").title() for category in EXPECTED_CATEGORIES]
    positions = list(range(len(categories)))
    bar_width = 0.19

    figure, axis = plt.subplots(figsize=(12, 7), dpi=180)
    figure.patch.set_facecolor("#f7f5ef")
    axis.set_facecolor("#f7f5ef")

    for offset, (key, label, color) in enumerate(METRIC_KEYS):
        values = [metrics_by_category[category][key] for category in EXPECTED_CATEGORIES]
        bars = axis.bar(
            [position + (offset - 1.5) * bar_width for position in positions],
            values,
            bar_width,
            label=label,
            color=color,
            edgecolor="#12343b",
            linewidth=0.5,
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, color="#12343b")

    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score", fontsize=11, color="#12343b")
    axis.set_xticks(positions, categories)
    axis.set_title(
        "EfficientAD Performance Metrics",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color="#12343b",
        pad=18,
    )
    axis.text(
        0,
        1.025,
        "Metrics from metrics/*.json",
        transform=axis.get_xaxis_transform(),
        fontsize=9,
        color="#587078",
    )
    axis.grid(axis="y", color="#d9dedb", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#9eaaa6")
    axis.tick_params(axis="x", colors="#12343b")
    axis.tick_params(axis="y", colors="#587078")
    axis.legend(frameon=False, loc="upper right", fontsize=9, ncol=2)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()