# Industrial Material Defect Detection

Real-time industrial material defect detection using **EfficientAD**, **Anomalib**, **MVTec AD**, and **OpenCV**.

The project supports:

- Training EfficientAD on an MVTec AD category
- Running inference on a single image
- Capturing images from a webcam
- Real-time webcam anomaly detection
- ROI-based inspection
- Camera-specific calibration using a known-good object
- Temporal filtering to reduce false positives

> **Current demo model:** MVTec AD `metal_nut`.

## Project structure

```text
industrial-defect-detection/
│
├── src/
│   ├── download_dataset.py
│   ├── train.py
│   ├── predict_image.py
│   ├── capture.py
│   └── realtime.py
│
├── models/
│   └── efficientad_metal_nut.ckpt
│
├── datasets/                 # local only, not committed
├── results/                  # local training/inference outputs, not committed
├── captured/                 # local webcam captures, not committed
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Important: what is and isn't included

### Included in Git

- Python source code in `src/`
- `README.md`
- `requirements.txt`
- `.gitignore`
- The trained EfficientAD checkpoint in `models/`

### Not included in Git

- MVTec AD dataset
- `.venv`
- Generated `results/`
- Captured images
- macOS `.DS_Store` files
- Python cache files

The MVTec dataset is downloaded locally by the setup script instead of being stored in this repository.

## Requirements

This project was developed/tested on Apple Silicon macOS with Python 3.11.

You need:

- Python 3.11
- Git
- Git LFS
- A webcam for real-time inspection

Git LFS is recommended for the `.ckpt` model file because model checkpoints are binary files and may be too large for normal Git.

## 1. Clone the repository

```bash
git clone https://github.com/giri5hsharma/Industrial-Material-Testing.git
cd industrial-defect-detection
```

## 2. Install Git LFS

On macOS with Homebrew:

```bash
brew install git-lfs
git lfs install
```

## 3. Create the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify Anomalib:

```bash
python -c "import anomalib; print(anomalib.__version__)"
```

Expected:

```text
2.6.2
```

Verify Apple MPS:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

On a compatible Apple Silicon machine this should print:

```text
MPS available: True
```

## 5. Download MVTec AD

The dataset is intentionally not stored in Git.

Run:

```bash
python src/download_dataset.py
```

The dataset will be created under:

```text
datasets/MVTecAD/
```

For the current model, the relevant category is:

```text
datasets/MVTecAD/metal_nut/
```

The important training split is:

```text
metal_nut/
└── train/
    └── good/
```

The test split contains good samples and multiple defect types.

## 6. Train EfficientAD

The current training script uses the MVTec `metal_nut` category.

Run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py
```

Training outputs are written under:

```text
results/
```

`results/` is intentionally ignored by Git.

If you create a new checkpoint that you want to distribute with the repository, copy it into `models/`.

Example:

```bash
cp results/EfficientAd/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt    models/efficientad_metal_nut.ckpt
```

## 7. Single-image inference

The current `predict_image.py` uses a configured checkpoint path and image path.

Set:

```python
MODEL_PATH = "models/efficientad_metal_nut.ckpt"
```

and choose the test image:

```python
IMAGE_PATH = "datasets/MVTecAD/metal_nut/test/scratch/000.png"
```

Then run:

```bash
python src/predict_image.py
```

This prints the anomaly score and predicted label.

## 8. Test the webcam

Before running anomaly detection, verify camera access:

```bash
python src/capture.py
```

macOS may ask for camera permission. If necessary, enable camera access under:

```text
System Settings
→ Privacy & Security
→ Camera
```

## 9. Real-time detection

The real-time application uses the committed model:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/realtime.py   --checkpoint "./models/efficientad_metal_nut.ckpt"   --skip 2
```

### Controls

| Key | Action |
|---|---|
| `D` | Toggle anomaly detection ON/OFF |
| `C` | Calibrate with a known-good object |
| `S` | Save the current camera frame |
| `Q` | Quit |

### Recommended workflow

1. Start the application.
2. Press `D` to enable detection.
3. Place a known-good metal nut inside the green inspection ROI.
4. Press `C`.
5. Keep the object and camera stable during calibration.
6. After calibration, test good and defective samples.

The calibration step measures the anomaly-score distribution produced by the actual camera/setup and creates a local detection threshold.

The real-time application also requires multiple anomalous frames before declaring a defect, reducing one-frame false positives.

## 10. Changing the inspection area

The real-time script contains the ROI settings:

```python
ROI_X = 170
ROI_Y = 70
ROI_W = 300
ROI_H = 300
```

Adjust these values to place the green inspection region around the object.

Only this ROI is sent to EfficientAD.

## 11. Performance

For higher camera/display FPS, the application supports frame skipping:

```bash
--skip 2
```

means inference is performed on every second frame.

For a lighter inference load:

```bash
--skip 3
```

The camera stream remains live while inference runs less frequently.

## 12. Model limitations

This repository currently uses an EfficientAD model trained on the **MVTec AD `metal_nut` category**.

It should therefore be treated as a demonstration of industrial anomaly detection, not as a universal defect detector for arbitrary materials.

For a real deployment:

1. Collect normal images using the actual inspection camera.
2. Match camera distance, lighting, background, and object positioning.
3. Retrain or adapt the anomaly detector using the target industrial component.
4. Validate thresholds on representative production samples.

## 13. Re-training from scratch

To train a fresh model:

```bash
python src/download_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py
```

The generated checkpoint will appear in the Anomalib results directory.

Copy the checkpoint you want to distribute to:

```text
models/
```

and update the README command if you rename it.

## License

Add your chosen project license here, for example MIT, before publishing if you intend the repository to be open source.
