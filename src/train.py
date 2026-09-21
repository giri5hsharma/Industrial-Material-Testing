from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def main():
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="metal_nut",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    model = EfficientAd(
        model_size="small",
        lr=1e-4,
    )

    engine = Engine(
        max_epochs=20,
        accelerator="mps",
        devices=1,
    )

    engine.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
