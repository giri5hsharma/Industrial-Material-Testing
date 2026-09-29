from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd


MODEL_PATH = "results/EfficientAd/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt"
IMAGE_PATH = "datasets/MVTecAD/metal_nut/test/good/000.png"


def main():
    model = EfficientAd()

    engine = Engine(
        accelerator="mps",
        devices=1,
    )

    dataset = PredictDataset(
        path=Path(IMAGE_PATH),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=MODEL_PATH,
    )

    for prediction in predictions:
        print("Image:", prediction.image_path)
        print("Score:", float(prediction.pred_score))
        print("Label:", int(prediction.pred_label))

        if int(prediction.pred_label) == 1:
            print("RESULT: ❌ DEFECT")
        else:
            print("RESULT: ✅ NORMAL")

if __name__ == "__main__":
    main()
