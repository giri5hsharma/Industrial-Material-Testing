# VIVA CHEATSHEET — one page

## Problem
Industrial defect detection as **one-class anomaly detection**. Train on normal only; detect any deviation. Dataset: **MVTec AD** (15 categories local; 9 trained; **6 deployed**: bottle, tile, leather, metal_nut [tier-1], wood, cable [tier-2]).

## Stack [CONFIG]
anomalib 2.6.2 (implements EfficientAD — model NOT hand-written in this repo), torch 2.14, Python 3.11, Flask app, OpenCV, Apple Silicon MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

## Architecture [LIB] (anomalib `efficient_ad/torch_model.py`)
- **Teacher**: PDN-Small, 4 convs + 2 avgpool, 384 out ch, **pretrained + frozen**, no_grad. Input 256² → output 64×64×384.
- **Student**: same PDN but **768 ch**: [0:384] imitates teacher, [384:768] imitates autoencoder. Trained on normal only → fails on anomalies.
- **Autoencoder**: encoder → 64-d bottleneck → decoder → 384 ch; imitates teacher features; catches **logical** defects (misplacement).
- Teacher features normalized by channel **mean/std** (computed on train set, stored in ckpt).

## Training [CODE + LIB]
`python src/train_all.py --epochs 20` → per category: MVTecAD datamodule (resize 256, **batch 1 — enforced**), Adam(lr 1e-4, wd 1e-5) on **student+AE only**, StepLR ×0.1 @95%, MPS.
**Loss = L_st + L_ae + L_stae**:
- L_st = hard loss (mean of top 0.1% of (T−S)²) + penalty (mean of S(ImageNet)²)
- L_ae = mean((T−AE)²) on augmented image
- L_stae = mean((AE − S[:,384:])²)
Validation good images → map quantiles (90%, 99.5%) stored for normalization.

## Inference [CODE] (`src/app.py`)
BGR→RGB → /255 → resize 256 → `[1,3,256,256]` → `model.model(x)`:
- map_st = mean_ch(T−S[:384])² ; map_stae = mean_ch(AE−S[384:])²
- upsample to 256², quantile-normalize, **fuse 0.5/0.5**
- **score = max(fused map)**, then anomalib min-max normalization → scores cluster ≈0.5
- **verdict: score ≥ registry threshold**

## Thresholds [CODE] — three, know all
1. Registry/web app: `good_max + 0.4·(defect_median − good_max)` (fallback good_max+0.01), **tuned on test set — acknowledged weakness**.
2. Anomalib internal F1-adaptive (validation) — used by engine.test.
3. realtime.py (**LEGACY webcam script, not the product**): `mean + 3σ` over 30 webcam frames, min 0.45; + temporal vote **5 of 7** frames.

Registry: bottle .5068, tile .5010, leather .5100, metal_nut .5225, wood .5347, cable .5100.

## Results [OUTPUT]
| cat | img_AUROC | img_F1 | px_AUROC | px_F1 |
|---|---:|---:|---:|---:|
| bottle | 1.000 | 0.992 | 0.979 | 0.774 |
| tile | 0.999 | 0.982 | 0.892 | 0.712 |
| leather | 0.990 | 0.968 | 0.975 | 0.595 |
| metal_nut | 0.973 | 0.957 | 0.948 | 0.768 |
| wood | 0.949 | 0.934 | 0.858 | 0.520 |
| cable | 0.928 | 0.863 | 0.971 | 0.639 |

Dropped: screw 0.862, transistor 0.778, capsule 0.688 (F1 0.911 = threshold artifact).
Live check [OUTPUT]: metal_nut good=0.496, bent=0.545 (DEFECT), scratch=0.493 (miss — known limit).

## Traps
- Score ≠ probability. AUROC ≠ accuracy. Heatmap red ≠ defect (per-image percentile stretch). MPS ≠ CUDA. Teacher never trains (only its mean/std come from your data). Thresholds are per-category. Model fails cross-category (metal_nut model + cable image → garbage).

## Key files
- `src/train_all.py` (train+eval+save), `src/app.py` (web inference — PRIMARY), `src/calibrate_thresholds.py` (thresholds), `models/registry.json`, `metrics/*.json`
- `src/realtime.py` = LEGACY unmaintained webcam experiment, metal_nut only. **Project is NOT real-time**: model-only 39 ms/image (~25 FPS ceiling) on MPS, but voting lag + frame skipping mean the webcam demo is not a real-time deployment.
- Model internals: `.venv/lib/python3.11/site-packages/anomalib/models/image/efficient_ad/torch_model.py` (losses L602-650, distance L582-600, score L577-580)

## Biggest weaknesses (say them first, disarm examiner)
1. Thresholds + validation stats touch the **test set** (leakage → optimistic metrics).
2. `predict_image.py` broken (hardcoded missing ckpt path).
3. 20 epochs undertrains hard categories (transistor/screw/capsule dropped).
4. Faint-scratch misses on metal_nut; pixel_F1 modest on wood/cable.
5. No AUPRO, no per-image latency benchmark, no seeds.

## 30-second pitch
"One-class industrial anomaly detection. EfficientAD per MVTec category trained on defect-free images only: frozen pretrained teacher, student learns to imitate it, autoencoder catches logical defects. Where imitation fails → anomaly map; max = score; per-category calibrated threshold → NORMAL/DEFECT in a Flask web app with heatmaps. Six categories deployed at 0.93–1.00 image AUROC."
