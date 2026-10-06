# Industrial Material Defect Detection

An industrial anomaly detection system built using **EfficientAD**, **Anomalib**, **PyTorch**, **MVTec AD**, and **OpenCV**.

The project detects defects in industrial materials using a combination of:

- **Global anomaly detection** using an autoencoding/reconstruction branch
- **Local anomaly detection** using a Teacher–Student feature-distillation branch
- **Image-level anomaly scoring** for GOOD / DEFECT classification
- **Pixel-level anomaly maps** for localizing defective regions
- **AUROC and F1-score** for quantitative evaluation
- A **Flask web application** for image-based inspection and visualization

The primary interface allows a user to upload an image, select the material category, and receive:

1. A **GOOD / DEFECT** prediction
2. An **anomaly score**
3. A **calibrated decision threshold**
4. An **anomaly heatmap** showing suspicious regions

---

# 1. Project Overview

Traditional supervised defect detection requires images containing every possible type of defect.

Industrial anomaly detection approaches the problem differently.

Instead of explicitly learning every defect class, the model is primarily trained using **defect-free (GOOD) samples** and learns what a normal object looks like.

During inference, an input image is compared against the learned representation of normality.

If the input differs significantly from what the model considers normal, a high anomaly score is produced.

The project uses **EfficientAD**, an efficient industrial anomaly detection architecture designed for fast detection while maintaining good anomaly localization performance.

---

# 2. High-Level Architecture

The system can be viewed as two complementary anomaly detection mechanisms:

```text
                         Input Image
                              │
                              ▼
                    ┌───────────────────┐
                    │ Image Preprocess  │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      ┌──────────────────┐          ┌─────────────────────┐
      │ Local Anomaly    │          │ Global Anomaly      │
      │ Detection        │          │ Detection            │
      │                  │          │                     │
      │ Teacher-Student  │          │ Autoencoder /       │
      │ Feature          │          │ Reconstruction      │
      │ Distillation     │          │ Branch              │
      └────────┬─────────┘          └──────────┬──────────┘
               │                               │
               ▼                               ▼
       Feature Difference              Reconstruction
          / Distillation                  Difference
               │                               │
               └──────────────┬────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Anomaly Map /     │
                    │ Anomaly Score     │
                    └─────────┬─────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
        Image-level result            Pixel-level result
        GOOD / DEFECT                 Defect localization
                │                            │
                └─────────────┬──────────────┘
                              ▼
                     Web Application
```

The two branches capture different kinds of anomalies.

### Local anomalies

The **Teacher–Student branch** focuses on differences in learned feature representations.

It is particularly useful for localized defects such as:

- scratches
- dents
- small surface irregularities
- missing components
- local texture changes

### Global anomalies

The **autoencoding/reconstruction branch** captures larger-scale differences between the input and the learned normal representation.

It can help identify:

- structural differences
- unusual shapes
- large damaged regions
- global appearance changes

The final anomaly representation combines information from these mechanisms.

---

# 3. EfficientAD

EfficientAD is designed specifically for industrial anomaly detection.

The important idea is that the model does not need to learn a separate supervised classifier for every possible defect.

Instead, it learns the distribution of **normal industrial images**.

Conceptually:

```text
GOOD training images
        │
        ▼
Learn representation of normality
        │
        ▼
Compare new image against normal representation
        │
        ▼
Large difference → likely anomaly
Small difference → likely normal
```

This makes the approach particularly useful in industrial environments where:

- defects are rare
- new defect types may appear
- collecting defective samples is difficult
- defect appearance is unpredictable

---

# 4. Teacher–Student Local Anomaly Detection

One component of EfficientAD uses a **Teacher–Student architecture**.

The Teacher network provides a learned feature representation of the input.

The Student network attempts to reproduce the Teacher's representation.

During training, the Student learns to match the Teacher on **normal images**.

Conceptually:

```text
                    Normal Image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          Teacher                Student
              │                     │
              ▼                     ▼
       Teacher Features       Student Features
              │                     │
              └──────────┬──────────┘
                         ▼
                  Feature Difference
```

For normal images:

```text
Teacher feature ≈ Student feature
        ↓
small difference
        ↓
low anomaly score
```

For an anomalous region:

```text
Teacher feature ≠ Student feature
        ↓
large feature difference
        ↓
high anomaly score
```

This difference can be calculated spatially, producing an **anomaly map** rather than only one score for the entire image.

This is why the Teacher–Student branch is useful for **local anomaly detection**.

---

# 5. Global Anomaly Detection

The second major component uses an **autoencoding/reconstruction mechanism**.

The basic idea is:

```text
Input Image
     │
     ▼
  Encoder
     │
     ▼
Latent Representation
     │
     ▼
  Decoder
     │
     ▼
Reconstructed Image
```

The encoder compresses the image into a latent representation.

The decoder attempts to reconstruct the original image.

For a normal image:

```text
Original ≈ Reconstruction
```

Therefore:

```text
Reconstruction Error ≈ small
```

For an image containing an anomaly:

```text
Original ≠ Reconstruction
```

Therefore:

```text
Reconstruction Error ↑
```

The reconstruction difference provides information about regions that do not match the learned normal appearance.

This gives the system a second source of anomaly information that complements the Teacher–Student branch.

---

# 6. Combining Local and Global Information

The final anomaly representation is not based on a single measurement.

The system uses information from both:

```text
Teacher–Student feature difference
             +
Autoencoder / reconstruction difference
             │
             ▼
       Anomaly representation
             │
             ▼
      Image anomaly score
             +
       Pixel anomaly map
```

This distinction is important:

### Image-level anomaly detection

Answers:

> **"Is this image anomalous?"**

The anomaly information is aggregated into an overall image-level score.

That score is then compared against a calibrated threshold:

```text
score > threshold
        │
        ▼
      DEFECT
```

while:

```text
score ≤ threshold
        │
        ▼
       GOOD
```

### Pixel-level anomaly detection

Answers:

> **"Where is the anomaly?"**

Instead of reducing the entire image to one value, the model retains spatial anomaly information.

This produces an anomaly map:

```text
Input image
     │
     ▼
Anomaly Map
     │
     ▼
Heatmap Overlay
```

High-intensity regions represent areas considered more anomalous.

---

# 7. Training Strategy

The MVTec AD dataset is used for training and evaluation.

The important property of the dataset is that the training set primarily contains **GOOD samples**.

Example:

```text
metal_nut/
├── train/
│   └── good/
│       ├── 000.png
│       ├── 001.png
│       └── ...
│
└── test/
    ├── good/
    ├── bent/
    ├── color/
    ├── flip/
    ├── scratch/
    └── ...
```

The model therefore learns:

> "This is what a normal metal nut looks like."

Rather than:

> "Here are all the possible defects."

This is the core difference between anomaly detection and conventional supervised classification.

---

# 8. MVTec AD Categories

The project was evaluated across multiple MVTec AD categories.

The categories include:

```text
bottle
cable
capsule
leather
metal_nut
screw
tile
transistor
wood
```

Each category is treated as a separate anomaly detection problem.

For example:

```text
metal_nut model
      ↓
learns normal metal nuts

wood model
      ↓
learns normal wood samples

bottle model
      ↓
learns normal bottles
```

A model trained for one category should therefore not be expected to work reliably on a completely different category.

---

# 9. Model Evaluation

Two major evaluation levels are considered.

## Image-level evaluation

The model produces one anomaly score for an entire image.

This is used for the final:

```text
GOOD / DEFECT
```

decision.

Important metrics include:

- **Image AUROC**
- **Image F1-score**
- Precision
- Recall

---

## Pixel-level evaluation

The model can also produce spatial anomaly information.

This is useful for determining whether the model correctly identifies the location of a defect.

Pixel-level metrics can include:

- Pixel AUROC
- Pixel F1-score
- Pixel-level precision
- Pixel-level recall
- segmentation/localization metrics where applicable

Therefore, the project evaluates both:

```text
                    Model
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Image-level         Pixel-level
        detection          localization
             │                 │
             ▼                 ▼
      GOOD / DEFECT       Where is defect?
```

---

# 10. Why AUROC?

The project uses **AUROC (Area Under the Receiver Operating Characteristic Curve)** as an important evaluation metric.

AUROC measures how well the model ranks anomalous samples above normal samples across different decision thresholds.

Conceptually:

```text
                 High AUROC
                     │
                     ▼
        Good separation between
        GOOD and DEFECT samples
```

An AUROC close to:

```text
1.0 → excellent separation
0.5 → approximately random
```

AUROC is especially useful for anomaly detection because the anomaly threshold can be changed after training.

For example, instead of evaluating only:

```text
threshold = 0.5
```

AUROC evaluates model discrimination across a range of thresholds.

This makes AUROC a useful metric to examine before selecting the operational threshold.

---

# 11. F1-Score

F1-score combines:

- Precision
- Recall

using their harmonic mean:

```text
             Precision × Recall
F1 = 2 × -----------------------------
             Precision + Recall
```

F1 is useful when both false positives and false negatives matter.

However, unlike AUROC, F1 depends directly on the selected classification threshold.

Therefore:

```text
AUROC
   ↓
Measures ranking/separation capability

F1
   ↓
Measures classification performance
at a particular threshold
```

Both metrics are useful, but they answer different questions.

---

# 12. Experimental Results

The following image-level results were obtained from the trained models:

| Category | Image AUROC | Image F1 | Status |
|---|---:|---:|---|
| bottle | 1.000 | 0.992 | Tier 1 |
| tile | 0.999 | 0.982 | Tier 1 |
| leather | 0.990 | 0.968 | Tier 1 |
| metal_nut | 0.973 | 0.957 | Tier 1 |
| wood | 0.949 | 0.934 | Tier 2 |
| cable | 0.928 | 0.863 | Tier 2 |
| screw | 0.862 | 0.877 | Not selected |
| transistor | 0.778 | 0.667 | Not selected |
| capsule | 0.688 | 0.911 | Not selected |

### Model selection

The primary selection metric is **image AUROC**, since the main application requirement is distinguishing normal and defective images.

The current deployment-oriented models are therefore divided into:

### Tier 1

High-performing models suitable for the main demonstration:

```text
bottle
tile
leather
metal_nut
```

### Tier 2

Models retained for experimentation but with lower performance:

```text
wood
cable
```

### Not selected

Models requiring further experimentation:

```text
screw
transistor
capsule
```

A high F1-score alone is not sufficient for selecting a model because F1 depends on the chosen threshold. AUROC provides a broader view of the model's ability to separate normal and anomalous samples.

---

# 13. Threshold Calibration

The raw anomaly score does not automatically determine whether an image is defective.

A decision threshold is therefore calibrated for each category.

Conceptually:

```text
              Anomaly Score
                   │
      ┌────────────┼────────────┐
      │            │            │
     GOOD       THRESHOLD      DEFECT
```

Different material categories can have different score distributions.

Therefore, a single global threshold is not necessarily appropriate.

The project maintains category-specific thresholds in:

```text
models/registry.json
```

Example:

```json
{
    "metal_nut": {
        "checkpoint": "models/metal_nut/model.ckpt",
        "threshold": 0.5225
    }
}
```

This allows the web application to automatically select the appropriate checkpoint and threshold when a category is selected.

---

# 14. Web Application

The primary user interface is a Flask web application.

```text
User
 │
 ▼
Select MVTec Category
 │
 ▼
Upload Image
 │
 ▼
Flask API
 │
 ▼
Load category checkpoint
 │
 ▼
EfficientAD inference
 │
 ├───────────────┐
 ▼               ▼
Anomaly Score   Anomaly Map
 │               │
 ▼               ▼
Threshold       Heatmap
comparison      generation
 │               │
 └───────┬───────┘
         ▼
    Web Interface
         │
    ┌────┴─────┐
    ▼          ▼
 GOOD/DEFECT  Heatmap
```

Run the application:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Application workflow

1. Select the material category.
2. Upload an inspection image.
3. Press **Detect**.
4. The selected model performs inference.
5. The anomaly score is calculated.
6. The score is compared against the category-specific threshold.
7. The application displays:
   - GOOD / DEFECT
   - anomaly score
   - threshold
   - anomaly heatmap

---

# 15. API

The Flask application also exposes API endpoints.

### List available categories

```bash
curl http://127.0.0.1:5000/categories
```

### Run prediction

```bash
curl -X POST \
  -F "category=metal_nut" \
  -F "file=@datasets/MVTecAD/metal_nut/test/bent/000.png" \
  http://127.0.0.1:5000/predict
```

Example response:

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

---

# 16. Project Structure

```text
industrial-defect-detection/
│
├── src/
│   ├── app.py
│   │   └── Flask web application
│   │
│   ├── templates/
│   │   └── index.html
│   │       └── Web interface
│   │
│   ├── train.py
│   │   └── Train EfficientAD for one category
│   │
│   ├── train_all.py
│   │   └── Train multiple MVTec categories
│   │
│   ├── evaluate.py
│   │   └── Evaluate trained models
│   │
│   ├── calibrate_thresholds.py
│   │   └── Generate category-specific thresholds
│   │
│   ├── predict_image.py
│   │   └── Single-image CLI inference
│   │
│   ├── download_dataset.py
│   │   └── Download MVTec AD
│   │
│   ├── capture.py
│   │   └── Legacy webcam frame capture
│   │
│   ├── camera_test.py
│   │   └── Legacy camera testing
│   │
│   └── realtime.py
│       └── Legacy real-time detection
│
├── models/
│   ├── registry.json
│   │   └── Checkpoint paths, thresholds and model metadata
│   │
│   └── <category>/
│       └── model.ckpt
│
├── metrics/
│   └── <category>.json
│       └── Evaluation results
│
├── datasets/
│   └── MVTecAD/
│       └── Local dataset — not committed
│
├── results/
│   └── Training outputs — not committed
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 17. Installation

The project was developed and tested on **Apple Silicon macOS**.

Recommended environment:

```text
Python 3.11
PyTorch
Anomalib 2.6.2
OpenCV
Flask
Git LFS
```

---

## 17.1 Install Git LFS

The trained checkpoints are binary files and are therefore stored using Git LFS.

Install Git LFS before cloning:

```bash
brew install git-lfs
git lfs install
```

---

## 17.2 Clone the Repository

```bash
git clone https://github.com/giri5hsharma/Industrial-Material-Testing.git
cd Industrial-Material-Testing
```

Verify that the checkpoints were downloaded:

```bash
ls -lh models/*/model.ckpt
```

If the checkpoint files are only around ~100 bytes, they are probably Git LFS pointer files.

Run:

```bash
git lfs install
git lfs pull
```

---

# 18. Create Virtual Environment

The project uses Python 3.11.

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

# 19. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
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

Check Apple MPS:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Expected on a compatible Apple Silicon system:

```text
MPS available: True
```

---

# 20. Download MVTec AD

The dataset is intentionally excluded from Git.

Download it using:

```bash
python src/download_dataset.py
```

The resulting structure is:

```text
datasets/
└── MVTecAD/
    ├── bottle/
    ├── cable/
    ├── capsule/
    ├── leather/
    ├── metal_nut/
    ├── screw/
    ├── tile/
    ├── transistor/
    └── wood/
```

The dataset is only required if you want to:

- retrain the models
- evaluate models
- recalibrate thresholds
- reproduce the experiments

The web application can run using the trained checkpoints without the full dataset.

---

# 21. Training

### Train all configured categories

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --epochs 20
```

### Train a single category

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py \
  --category metal_nut \
  --epochs 20
```

Training outputs are stored under:

```text
results/
```

The final checkpoint is copied to:

```text
models/<category>/model.ckpt
```

Evaluation metrics are written to:

```text
metrics/<category>.json
```

---

# 22. Recalibrate Thresholds

After training:

```bash
python src/calibrate_thresholds.py
```

This generates category-specific anomaly thresholds.

The thresholds are stored in:

```text
models/registry.json
```

---

# 23. Single Image Inference

For command-line inference, configure:

```python
MODEL_PATH = "models/metal_nut/model.ckpt"
IMAGE_PATH = "datasets/MVTecAD/metal_nut/test/scratch/000.png"
```

Then:

```bash
python src/predict_image.py
```

The script outputs the anomaly score and predicted result.

---

# 24. Retraining a Specific Category

For example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py \
  --category screw \
  --epochs 40 \
  --force
```

The `--force` option allows an existing model to be retrained.

This can be useful for categories with poor initial performance.

---

# 25. Why Some Categories Perform Better

MVTec categories have very different visual characteristics.

For example, some objects have:

- consistent geometry
- simple backgrounds
- clearly defined defect patterns

while others have:

- complex textures
- subtle defects
- high intra-class variation
- difficult boundaries between normal and anomalous regions

Therefore, the same EfficientAD configuration does not necessarily produce the same performance for every category.

This is why model evaluation and category-specific threshold calibration are important.

---

# 26. Limitations

This project is a demonstration of industrial anomaly detection using MVTec AD.

The trained models should **not** be interpreted as universal industrial defect detectors.

A model trained on:

```text
MVTec metal_nut
```

does not automatically generalize to arbitrary:

```text
industrial nuts
mechanical components
metal surfaces
factory products
```

Real-world deployment would require:

1. Collecting images using the actual inspection camera.
2. Matching production lighting.
3. Matching camera distance and viewpoint.
4. Controlling object positioning.
5. Collecting representative GOOD samples.
6. Validating the anomaly-score distribution.
7. Calibrating thresholds using production data.
8. Testing false positives and false negatives.
9. Retraining/adapting the model for the target component.

---

# 27. Legacy Real-Time Webcam Detection

The repository also contains an older OpenCV-based webcam pipeline.

These scripts are kept for backup/demo purposes and are **not the primary application**.

Relevant files:

```text
src/capture.py
src/camera_test.py
src/realtime.py
```

The real-time implementation includes:

- ROI-based inspection
- camera-specific calibration
- frame skipping
- temporal filtering
- anomaly detection using the metal_nut model

---

## Webcam Controls

| Key | Action |
|---|---|
| `D` | Toggle anomaly detection |
| `C` | Calibrate using a known-good object |
| `S` | Save current frame |
| `Q` | Quit |

Run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/realtime.py \
  --checkpoint "./models/metal_nut/model.ckpt" \
  --skip 2
```

The ROI is configured using:

```python
ROI_X = 170
ROI_Y = 70
ROI_W = 300
ROI_H = 300
```

Only the selected ROI is passed to the anomaly detector.

Frame skipping can be increased to reduce inference load:

```bash
--skip 3
```

---

# 28. Camera Calibration

The legacy webcam pipeline can calibrate the anomaly-score distribution using a known-good object.

The workflow is:

```text
Known-good object
       │
       ▼
Camera
       │
       ▼
Inspection ROI
       │
       ▼
EfficientAD
       │
       ▼
Normal anomaly-score distribution
       │
       ▼
Camera-specific threshold
```

This is useful because the same object can produce different anomaly scores under different:

- cameras
- lighting conditions
- distances
- backgrounds
- viewpoints

The web application instead uses the category-specific calibrated thresholds stored in `registry.json`.

---

# 29. Reproducibility

To reproduce the main experiment:

```bash
git clone https://github.com/giri5hsharma/Industrial-Material-Testing.git
cd Industrial-Material-Testing

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python src/download_dataset.py

PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train_all.py --epochs 20

python src/calibrate_thresholds.py
```

Then run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 30. Key Concepts Demonstrated

This project demonstrates several important concepts in modern computer vision and anomaly detection:

- Industrial anomaly detection
- Unsupervised / one-class learning
- EfficientAD
- Teacher–Student knowledge distillation
- Feature-space anomaly detection
- Autoencoder-based reconstruction
- Image-level anomaly detection
- Pixel-level anomaly localization
- Anomaly maps
- Threshold calibration
- AUROC
- F1-score
- Precision and Recall
- MVTec AD
- PyTorch
- Anomalib
- OpenCV
- Flask
- Apple Silicon MPS acceleration

---

# 31. Summary

The overall system can be summarized as:

```text
                  MVTec AD
                     │
                     ▼
              Normal Training
                     │
                     ▼
               EfficientAD
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Teacher–Student       Autoencoding /
     Distillation         Reconstruction
          │                     │
          ▼                     ▼
    Local anomalies       Global anomalies
          │                     │
          └──────────┬──────────┘
                     ▼
               Anomaly Map
                     │
                     ▼
              Anomaly Score
                     │
                     ▼
          Category Threshold
                     │
             ┌───────┴───────┐
             ▼               ▼
           GOOD            DEFECT
                             │
                             ▼
                     Heatmap / Localization
```

The primary objective is therefore not simply to classify images as defective.

The system attempts to answer **two questions simultaneously**:

> **Is this object anomalous?**

and

> **Where is the anomaly located?**

This combination of **image-level detection** and **pixel-level localization**, using both **global reconstruction information** and **local Teacher–Student feature discrepancies**, forms the core of the project.
