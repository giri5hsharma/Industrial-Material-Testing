# Industrial Material Defect Detection

Industrial material defect detection using **EfficientAD**, **Anomalib**, **MVTec AD**, and **OpenCV**.

The project supports:

- **Web app (primary): upload an image, pick an MVTec class, and get a GOOD / DEFECT verdict with an anomaly heatmap**
- Training EfficientAD on any MVTec AD category
- Running inference on a single image
- Legacy real-time webcam anomaly detection (kept as backup):
  - ROI-based inspection
  - Camera-specific calibration using a known-good object
  - Temporal filtering to reduce false positives

> **Current demo models:** trained EfficientAD checkpoints for the MVTec AD categories selected below.

## Project structure

```text
industrial-defect-detection/
│
├── src/
│   ├── app.py                  # Flask web app (upload & detect) — PRIMARY
│   ├── templates/
│   │   └── index.html          # Web UI
│   ├── calibrate_thresholds.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── train.py
│   ├── train_all.py
│   ├── predict_image.py
│   ├── capture.py              # LEGACY: webcam frame capture
│   ├── camera_test.py          # LEGACY: webcam permission test
│   └── realtime.py             # LEGACY: real-time webcam detection
│
├── models/
│   ├── registry.json           # per-class checkpoint + threshold
│   └── <category>/
│       └── model.ckpt          # trained EfficientAD model
│
├── metrics/                    # per-class evaluation metrics
├── datasets/                   # local only, not committed
├── results/                    # local training logs/history, not committed
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
- Trained EfficientAD checkpoints under `models/<category>/model.ckpt`
- Per-class metrics in `metrics/`
- The model registry `models/registry.json`

### Not included in Git

- MVTec AD dataset
- `.venv`
- Generated `results/`
- macOS `.DS_Store` files
- Python cache files

The MVTec dataset is downloaded locally by the setup script instead of being stored in this repository.

## Requirements

This project was developed/tested on Apple Silicon macOS with Python 3.11.

You need:

- Python 3.11
- Git
- Git LFS
- A webcam (only for the legacy real-time scripts)

Git LFS is recommended for the `.ckpt` model files because model checkpoints are binary files and may be too large for normal Git.

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

Each category follows the layout:

```text
<category>/
├── train/
│   └── good/
└── test/
    ├── good/
    └── <defect types>/
```

The training split contains only defect-free samples. The test split
contains good samples and multiple defect types.

## 6. Train EfficientAD

To train all configured categories:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --epochs 20
```

To train a single category:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --category metal_nut --epochs 20
```

Training logs are written under `results/` (ignored by Git).
The final checkpoint is copied to:

```text
models/<category>/model.ckpt
```

and evaluation metrics are written to:

```text
metrics/<category>.json
```

After training, regenerate the per-class thresholds:

```bash
python src/calibrate_thresholds.py
```

## 7. Web app: upload & detect (primary)

The primary way to use this project is the Flask web app: upload an
image, select the MVTec AD class, and get a GOOD / DEFECT verdict plus
an anomaly heatmap.

Run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Workflow:

1. Select the class using the radio buttons.
2. Drag & drop an image (or click to browse).
3. Press **Detect**.
4. The UI shows the verdict, the anomaly score, the calibrated threshold,
   and a heatmap overlay highlighting the most anomalous regions.

The class selector lists only the well-performing classes kept after the
metrics review. Tier-1 classes are marked "production-grade";
tier-2 classes are marked "experimental".

The API endpoints are also usable directly:

```bash
# list available classes
curl http://127.0.0.1:5000/categories

# predict
curl -X POST \
  -F "category=metal_nut" \
  -F "file=@datasets/MVTecAD/metal_nut/test/bent/000.png" \
  http://127.0.0.1:5000/predict
```

Response:

```json
{
  "category": "metal_nut",
  "tier": 1,
  "score": 0.5454,
  "threshold": 0.5225,
  "is_defect": true,
  "result": "DEFECT",
  "heatmap": "<base64 PNG>"
}
```

Note: very faint scratches can still score below the calibrated
threshold on `metal_nut` and `wood`.

## 8. Single-image inference (CLI)

`predict_image.py` runs one image through a checkpoint without the web UI.

Set:

```python
MODEL_PATH = "models/metal_nut/model.ckpt"
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

## 9. Model selection

All MVTec AD categories were trained with the same EfficientAD-small,
20-epoch setup. The following metrics were collected:

| Category | image_AUROC | image_F1 | Decision |
|---|---|---|---|
| bottle | 1.000 | 0.992 | KEEP tier-1 |
| tile | 0.999 | 0.982 | KEEP tier-1 |
| leather | 0.990 | 0.968 | KEEP tier-1 |
| metal_nut | 0.973 | 0.957 | KEEP tier-1 |
| wood | 0.949 | 0.934 | KEEP tier-2 (experimental) |
| cable | 0.928 | 0.863 | KEEP tier-2 (experimental) |
| screw | 0.862 | 0.877 | DROP |
| transistor | 0.778 | 0.667 | DROP |
| capsule | 0.688 | 0.911 | DROP |

Primary selection metric is **image_AUROC**: it measures how well the
model separates good images from defective images, which is exactly
what the GOOD/DEFECT verdict needs.

Dropped classes were removed from `models/`. They can be retrained with
more epochs or a different model if needed:

```bash
python src/train_all.py --category screw --epochs 40 --force
```

## 10. Model limitations

This repository currently serves EfficientAD models trained on the
**MVTec AD** categories listed above.

It should therefore be treated as a demonstration of industrial anomaly detection, not as a universal defect detector for arbitrary materials.

For a real deployment:

1. Collect normal images using the actual inspection camera.
2. Match camera distance, lighting, background, and object positioning.
3. Retrain or adapt the anomaly detector using the target industrial component.
4. Validate thresholds on representative production samples.

## 11. Re-training from scratch

To train all categories:

```bash
python src/download_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --epochs 20
```

To train a single category:

```bash
python src/download_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --category metal_nut --epochs 20
```

The generated checkpoints appear under `models/<category>/model.ckpt`
and evaluation metrics are written to `metrics/<category>.json`.

After training, regenerate the calibrated thresholds:

```bash
python src/calibrate_thresholds.py
```

and update `models/registry.json` if you want to add, drop, or re-tier
classes.

## 12. Legacy: real-time webcam detection (backup)

> **Legacy.** These scripts predate the upload-based web app and are
> kept as a backup for physical-inspection demos. They are not used by
> the web app and are no longer actively maintained. The webcam scripts
> are also hardcoded to the old single `metal_nut` workflow.

### Test the webcam

Before running anomaly detection, verify camera access:

```bash
python src/camera_test.py
```

macOS may ask for camera permission. If necessary, enable camera access under:

```text
System Settings
→ Privacy & Security
→ Camera
```

### Real-time detection

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/realtime.py \
  --checkpoint "./models/metal_nut/model.ckpt" \
  --skip 2
```

#### Controls

| Key | Action |
|---|---|
| `D` | Toggle anomaly detection ON/OFF |
| `C` | Calibrate with a known-good object |
| `S` | Save the current camera frame |
| `Q` | Quit |

#### Recommended workflow

1. Start the application.
2. Press `D` to enable detection.
3. Place a known-good metal nut inside the green inspection ROI.
4. Press `C`.
5. Keep the object and camera stable during calibration.
6. After calibration, test good and defective samples.

The calibration step measures the anomaly-score distribution produced by the actual camera/setup and creates a local detection threshold.

The real-time application also requires multiple anomalous frames before declaring a defect, reducing one-frame false positives.

### Changing the inspection area

The real-time script contains the ROI settings:

```python
ROI_X = 170
ROI_Y = 70
ROI_W = 300
ROI_H = 300
```

Adjust these values to place the green inspection region around the object.

Only this ROI is sent to EfficientAD.

### Performance

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

## License

Add your chosen project license here, for example MIT, before publishing if you intend the repository to be open source.
