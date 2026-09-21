from pathlib import Path

from anomalib.data import MVTecAD


DATASET_ROOT = Path("datasets/MVTecAD")
CATEGORY = "metal_nut"


def main():
    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    print("Preparing MVTec AD...")
    datamodule.prepare_data()

    print("Setting up datamodule...")
    datamodule.setup()

    print()
    print("Dataset ready.")
    print(f"Root: {DATASET_ROOT.resolve()}")
    print(f"Category: {CATEGORY}")
    print(f"Training samples: {len(datamodule.train_data)}")
    print(f"Test samples: {len(datamodule.test_data)}")


if __name__ == "__main__":
    main()
