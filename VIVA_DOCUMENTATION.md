# VIVA DOCUMENTATION — Industrial Defect Detection (Simple Language Version)

This version explains everything in plain English. Read this first. Use the other files (`VIVA_CHEATSHEET.md`, `VIVA_QUESTIONS.md`) after you understand this one.

Labels used:
- **[CODE]** = I checked this in your actual code files
- **[LIB]** = I checked this inside the anomalib library (installed in `.venv`)
- **[OUTPUT]** = I actually ran it and saw this result
- **[CONFIG]** = from config files like `requirements.txt` or `registry.json`
- **[PAPER]** = from the original EfficientAD research paper
- **[UNKNOWN]** = could not verify

---

## THE MOST IMPORTANT FACT — READ THIS FIRST

**You did not write the EfficientAD model yourself. The model comes from a library called `anomalib` (made by Intel), version 2.6.2.** [CODE]

Look in your `src/` folder: there is no neural network code there. No layers, no model class, nothing. All the ML brains live inside the anomalib library (in `.venv/lib/python3.11/site-packages/anomalib/`). [LIB]

**What YOU actually built:**

1. `train_all.py` — a script that trains one model per product category and saves it
2. `calibrate_thresholds.py` — a script that decides the "defect score cutoff" for each category
3. `models/registry.json` — a small config file listing which models are deployed and their cutoffs
4. `app.py` + `templates/index.html` — a website where you upload a photo and get NORMAL or DEFECT with a heatmap
5. `realtime.py` — an OLD webcam experiment (legacy, not maintained, not the real product)
6. The decision to keep 6 good categories and drop 3 bad ones (based on measured results)

**If the examiner asks "did you implement the paper?", say this honestly:**
> "The core model is Anomalib's implementation of the EfficientAD paper. My work is the training pipeline, the threshold calibration, the model registry, and the web app that serves the models. I also validated the behavior experimentally."

That answer is completely fine — BUT you still need to understand how the model inside anomalib works, because the examiner can ask about it. Everything important is explained below in simple terms.

---

# 1. WHAT THE PROJECT DOES (IN ONE MINUTE)

Imagine a factory making metal nuts. Sometimes a nut comes out bent or scratched. You want a camera system that says "this one is bad".

The normal way (classification) would need thousands of photos of every possible defect — bent, scratched, cracked, wrong color, broken... That's expensive and slow, and new defect types appear that you never photographed.

**Your project uses a different idea: anomaly detection.** Instead of learning what defects look like, the model learns **only what a GOOD product looks like**. Then anything that doesn't look normal gets flagged.

So:
- Training: show the model only perfect products (e.g. 200 photos of good metal nuts)
- Testing: show it a photo → it gives an **anomaly score** (a number) and an **anomaly map** (a heatmap showing WHERE the problem is)
- Decision: if the score is above a cutoff (threshold) → DEFECT, else NORMAL

You trained this for 6 product types from the MVTec AD dataset: bottle, tile, leather, metal_nut (these 4 work really well), wood and cable (these 2 work okay, labeled "experimental"). [CODE + OUTPUT]

---

# 2. REPOSITORY MAP (WHAT EVERY FILE DOES)

```text
industrial-defect-detection/
├── src/
│   ├── app.py                    # The website (MAIN way to use the project)
│   ├── train_all.py              # Trains models for all categories
│   ├── calibrate_thresholds.py   # Decides the DEFECT cutoff per category
│   ├── evaluate.py               # Re-tests one trained model
│   ├── realtime.py               # OLD webcam experiment — legacy, not the product
│   ├── train.py                  # Old single-category trainer (superseded)
│   ├── predict_image.py          # Single-image test — BROKEN (bad file path)
│   ├── generate_metrics_chart.py # Makes the bar chart of results
│   ├── download_dataset.py       # Downloads/prepares the dataset
│   ├── capture.py                # Webcam photo capture (trivial)
│   ├── camera_test.py            # Webcam test (trivial)
│   └── templates/index.html      # The web page UI
├── models/
│   ├── registry.json             # List of deployed models + their cutoffs
│   └── bottle/model.ckpt etc.    # 6 trained model files (~72 MB each)
├── metrics/                      # Result files (one JSON per category) + charts
├── datasets/MVTecAD/             # The dataset (15 categories, kept locally)
├── results/                      # Old training run outputs (images only, no weights)
└── requirements.txt              # Says: anomalib 2.6.2, flask, opencv, matplotlib
```

### The files that matter most for the viva:

| File | Why it matters | Importance |
|---|---|---|
| `src/train_all.py` | The whole training pipeline | HIGH |
| `src/app.py` | The web app = how the model is actually used | HIGH |
| `src/calibrate_thresholds.py` | How the DEFECT/NORMAL cutoff is chosen | HIGH |
| `models/registry.json` | Deployed models + cutoffs | HIGH |
| `metrics/*.json` | Your results | HIGH |
| `src/realtime.py` | Old webcam script — only if asked about real-time | MEDIUM |

---

# 3. THE DATASET (MVTec AD) — SIMPLE EXPLANATION

**MVTec AD** is the standard test dataset for industrial anomaly detection. It has 15 categories (bottle, cable, capsule, wood, tile, transistor, etc.). You have all 15 downloaded locally, trained 9, and deployed 6. [CODE]

Each category folder looks like this:

```text
metal_nut/
├── train/
│   └── good/           # ~200 photos of PERFECT metal nuts (nothing else!)
├── test/
│   ├── good/           # more perfect nuts (for testing)
│   ├── bent/           # defective nuts (bent)
│   ├── scratch/        # defective nuts (scratched)
│   └── ...             # other defect types
└── ground_truth/
    └── bent/000_mask.png   # black-white images marking WHERE the defect is
```

Three things to memorize:

1. **Training uses only `train/good/`.** The model never sees a single defect during training. This is called **one-class learning**. [CODE]
2. **Defects only exist in `test/`** — used to check if the model works.
3. **The masks in `ground_truth/` are only for evaluation** — they let us measure how good the heatmap is, never for training.

**Why one model per category?** Because "normal" for a bottle looks nothing like "normal" for a cable. One shared model would have to learn 6 completely different ideas of normal — it becomes too loose and starts accepting defects. Separate models are sharper. The cost: 6 model files and the user must pick the category in the web app.

**Common viva question: "What happens if I show the metal_nut model a photo of a cable?"**
Answer: garbage. The cable looks nothing like any training image, so the model panics — high anomaly score everywhere, meaningless heatmap. That's exactly why the web app forces you to select the category first.

---

# 4. THE MODEL — EFFICIENTAD IN SIMPLE TERMS

EfficientAD is an anomaly detection method from a 2023 research paper (the name means it's designed to be *fast* — factories need answers in milliseconds). [PAPER]

It has **three parts**. Here's the easy analogy:

### Part 1: The Teacher (frozen expert)
- A small neural network that was already trained by the anomalib authors on general images.
- It converts a photo into a compact "description" — a grid of numbers (a **feature map**) saying what each small patch of the image looks like.
- **Frozen** = it never changes during your training. Think of it as an experienced inspector whose judgment you trust completely. [LIB]

### Part 2: The Student (the apprentice)
- A copy of the teacher's network, but untrained.
- During YOUR training, it learns to produce the **same descriptions as the teacher** — but it only ever sees your **good products** (e.g. perfect metal nuts).
- So the student becomes an expert at describing **good metal nuts only**.

**The trick:** show the system a good nut → teacher and student agree (similar descriptions) → low difference → NORMAL. Show it a defective nut → the teacher still describes it fine (it knows general images), but the student has no idea what to do with a bent/scratched region — it never learned that → its description is **wrong** → **big difference** → DEFECT.

**The disagreement between teacher and student IS the anomaly signal.** That's the whole core idea. [LIB]

### Part 3: The Autoencoder (the global checker)
- The teacher/student compare small patches (~33×33 pixels each). But some defects are **logical**, not local: e.g. a transistor placed in the wrong position, a missing part. Every individual patch looks normal, but the overall arrangement is wrong.
- The autoencoder squeezes the whole image through a tiny bottleneck (64 numbers), so it can only remember the **global layout**. It learns to reproduce the teacher's description for normal layouts.
- The student's second half learns to copy the autoencoder. Disagreement here = logical anomaly.
- Final heatmap = average of (teacher-vs-student difference) and (autoencoder-vs-student difference). 50/50. [LIB]

**Simple summary:** three networks. One frozen expert (teacher), one apprentice trained only on good products (student), one global-layout checker (autoencoder). Defect = where they disagree.

### What happens if you remove a part?
- Remove teacher → nothing to imitate → everything collapses.
- Remove student → no learned comparison → no detection.
- Remove autoencoder → still detects scratches/cracks (texture defects) but gets worse at "misplaced/missing part" defects.

---

# 5. TRAINING — WHAT ACTUALLY HAPPENS

Command: `python src/train_all.py --epochs 20` [CODE]

For each category (bottle, cable, etc.), the script does:

1. **Load the dataset** — all `train/good` images, resized to 256×256, one image at a time (batch size 1 — the method requires this; anomalib literally raises an error otherwise). [LIB]
2. **Build the model** — `EfficientAd(model_size="small", lr=1e-4)`.
3. **Setup before training** (anomalib does this automatically): [LIB]
   - Downloads the pretrained teacher weights
   - Downloads a small ImageNet dataset (used for the "penalty" — explained below)
   - Computes the teacher's average output statistics on YOUR training images (used to normalize the teacher's outputs so all channels matter equally)
4. **Training loop, 20 epochs.** For each good image, the model computes a **loss** (a number meaning "how badly is the student imitating?") and an optimizer (Adam) nudges the student + autoencoder weights to reduce it. The teacher is never touched. [LIB]
5. **Validation step** — computes reference statistics (quantiles) from good validation images, stored inside the model. Used later to scale the heatmaps. [LIB]
6. **Test** — runs the model on the test set and computes 4 metrics (see §9). [CODE]
7. **Save** — copies the best checkpoint to `models/<category>/model.ckpt` and writes results to `metrics/<category>.json`. [CODE]

### The loss (simple version)

The training loss has 3 pieces, added together: [LIB]

1. **Imitation loss:** "student, copy the teacher's output." But not on the whole image equally — it focuses on the **0.1% worst patches** (the patches where the student's copy is currently worst). This is called *hard example mining* — like a teacher focusing lessons only on the topics you're worst at.
2. **Penalty loss:** "student, when shown random ImageNet photos (not metal nuts), output near-zero." This stops the student from learning to imitate the teacher on random stuff — so that when a real defect appears (also "random stuff" to the student), the student does NOT copy the teacher correctly, and the difference stays big.
3. **Autoencoder losses:** teach the autoencoder to reproduce the teacher's description, and teach the student's second half to reproduce the autoencoder's output.

Total loss = piece 1 + piece 2 + piece 3. Plain sum, no weighting. [LIB]

**Optimizer:** Adam, learning rate 0.0001, 20 epochs. Near the end (95% of training), the learning rate drops 10× for fine-tuning. [LIB]

---

# 6. INFERENCE — WHAT HAPPENS WHEN YOU UPLOAD A PHOTO

This is `src/app.py`, function `predict()` (lines 116–130). [CODE]

Step by step, with a real example:

1. **Read the image** with OpenCV → a grid of pixels, e.g. 900×900×3 (height × width × RGB).
2. **Convert BGR→RGB** (OpenCV loads colors in BGR order; the model expects RGB — easy bug source!).
3. **Rearrange** to 3×900×900 (channels first — PyTorch format) and divide by 255 → numbers between 0 and 1.
4. **Resize to 256×256** and add batch dimension → final tensor shape **1×3×256×256** (1 image, 3 color channels, 256×256 pixels).
5. **Forward pass** — the model internally:
   - Teacher describes the image → grid **384×64×64** (384 numbers per patch, 64×64 patches).
   - Student produces its version → **768×64×64** (double: first 384 channels copy the teacher, last 384 copy the autoencoder).
   - Autoencoder produces its version → 384×64×64.
   - Difference 1: teacher minus student's first half → **the texture-anomaly map**
   - Difference 2: autoencoder minus student's second half → **the logical-anomaly map**
   - Average the two maps (50/50), blow it back up to 256×256 → **the anomaly map** (one "suspiciousness" value per image region).
   - **Anomaly score = the single highest value in the map** (the max). The logic: if even one patch is very anomalous, the product is defective. [LIB]
6. **Post-processing** — anomalib rescales the score using statistics from validation, which is why all your scores land around 0.5. [LIB + OUTPUT]
7. **Decision** — `app.py` line 247: `is_defect = score >= threshold`. The threshold comes from `registry.json` for the selected category.
8. **Heatmap** — the anomaly map is colored (red = most anomalous) and blended over the photo for the user.

**Real numbers I measured on your machine** [OUTPUT] (metal_nut model):
- Good nut → score **0.496** → below threshold 0.5225 → NORMAL ✓
- Bent nut → score **0.545** → above threshold → DEFECT ✓
- Faint scratch → score **0.493** → below threshold → NORMAL ✗ (a miss!)

That last one matters: the README itself warns that faint scratches on metal_nut can be missed. If the examiner asks about failures, this is your honest, verified example.

---

# 7. THE ANOMALY SCORE — CLEARING UP CONFUSION

Things you MUST be able to say:

- **The score is NOT a probability.** 0.545 does not mean "54.5% chance of defect". It's a rescaled measurement of "how much the student disagreed with the teacher, in the worst patch of the image". [LIB]
- **High score** = at least one image region looks very abnormal to the model.
- **Low score** = every region matched what the model learned as normal.
- **Scores are only comparable within the same category.** A 0.52 on bottle and a 0.52 on cable come from different models with different scales. This is why each category has its own threshold.
- **Why is the score a max and not an average?** Because defects are small. If you averaged the map, a tiny scratch's signal would drown in thousands of normal values. The max preserves it. The cost: one noisy pixel (a specular reflection, camera noise) can spike the score.

---

# 8. THE THRESHOLD — THE WEAKEST PART (BE HONEST ABOUT IT)

The threshold is the cutoff: score above → DEFECT, below → NORMAL. Your project actually has THREE different thresholds: [CODE]

1. **The web app threshold** (the important one): `calibrate_thresholds.py` computes it like this:
   - Run the model on all test images of a category
   - Find the highest score among GOOD images (`good_max`)
   - Find the median score among DEFECT images (`defect_median`)
   - `threshold = good_max + 0.4 × (defect_median − good_max)` — put the cutoff 40% of the way from "worst good" to "typical defect"
2. **Anomalib's internal threshold** — computed automatically during training on validation data; used for the F1 metrics, not by your app.
3. **The old webcam script's threshold** — mean + 3×std of 30 frames of a known-good object in front of the camera.

Your deployed thresholds [CONFIG]:

| category | threshold |
|---|---:|
| bottle | 0.5068 |
| tile | 0.5010 |
| leather | 0.5100 |
| metal_nut | 0.5225 |
| wood | 0.5347 |
| cable | 0.5100 |

**Why this is weak (say this if asked — it turns a weakness into a strength):**
1. The thresholds were tuned **on the test set** — the same data used to report results. That's a form of data leakage. The proper way: tune on a separate validation set.
2. The "0.4" is a hand-picked number, not the result of optimization. The proper way: pick the threshold from a ROC curve (e.g. Youden's J) or a target false-alarm rate.

**Tradeoff to memorize:** lower threshold → catch more defects but more false alarms; higher threshold → fewer false alarms but more missed defects. In quality control, missed defects are usually worse.

---

# 9. THE METRICS — WHAT YOUR RESULTS MEAN

Your project computes 4 metrics per category [CODE] (in `train_all.py` / `evaluate.py`):

**image_AUROC** (the headline number)
- Meaning: pick one random defect photo and one random good photo. What's the chance the defect photo got the higher score? AUROC = that probability.
- 1.0 = perfect ranking. 0.5 = coin flip (useless).
- Big advantage: it does NOT depend on any threshold.

**image_F1Score**
- F1 = harmonic mean of precision and recall at a specific threshold: `2·P·R/(P+R)`.
- Precision = of all images you called DEFECT, how many really were? Recall = of all real defects, how many did you catch?
- **Warning:** F1 depends on the threshold, so it can be misleading. Real example from your project: the capsule model scored F1 = 0.911 but AUROC = 0.688. How? It basically called almost everything "defect" → caught most defects (high recall) → decent F1, while its actual ranking ability (AUROC) was poor. This is why model selection used AUROC, not F1. [CONFIG]

**pixel_AUROC** — same idea as image AUROC but per pixel, using the ground-truth masks. Measures heatmap quality (did it highlight the right place?).

**pixel_F1Score** — F1 over pixels. Always lower than image_F1, because getting the verdict right is much easier than highlighting the exact defect boundary.

Your results [OUTPUT]:

| category | image_AUROC | image_F1 | pixel_AUROC | pixel_F1 | verdict |
|---|---:|---:|---:|---:|---|
| bottle | 1.000 | 0.992 | 0.979 | 0.774 | tier-1 |
| tile | 0.999 | 0.982 | 0.892 | 0.712 | tier-1 |
| leather | 0.990 | 0.968 | 0.975 | 0.595 | tier-1 |
| metal_nut | 0.973 | 0.957 | 0.948 | 0.768 | tier-1 |
| wood | 0.949 | 0.934 | 0.858 | 0.520 | tier-2 |
| cable | 0.928 | 0.863 | 0.971 | 0.639 | tier-2 |

Dropped (too weak): screw (0.862), transistor (0.778), capsule (0.688). [CONFIG]

**Metrics NOT computed:** accuracy, precision, recall (standalone), AUPRO, IoU/Dice. If asked about AUPRO (the paper's localization metric), say honestly it's not computed — pixel_AUROC is the localization proxy instead.

---

# 10. REAL-TIME — IMPORTANT: YOUR PROJECT IS NOT REAL-TIME

**Do not present this project as a real-time system.** The real product is the upload-based web app. `realtime.py` is an OLD webcam experiment — the README itself marks it "legacy / backup / no longer maintained", and it only works with metal_nut. [CODE + CONFIG]

I measured the actual speed [OUTPUT]:
- The model itself is fast: **39 ms per image** on your Mac (MPS) → about 25 images/second if you only count the model.
- But the webcam script adds frame skipping (only every Nth frame is analyzed) and a voting system (needs 5 out of 7 recent frames to agree before saying DEFECT), so the actual verdict is delayed. Fast model ≠ real-time product.

If asked "is it real-time?", answer:
> "No. The product is an upload-based inspection app. The model itself runs at ~39 ms per image on my Mac, but the webcam script is an unmaintained legacy demo with voting lag, not a real-time deployment."

(The webcam script's ideas are still worth understanding if asked: it calibrates its threshold live — mean + 3×std over 30 frames of a known-good object — because webcam images look different from the dataset, and it votes over 7 frames to avoid one-frame false alarms.)

---

# 11. HARDWARE — APPLE SILICON / MPS

- **MPS** = Metal Performance Shaders — how PyTorch uses the GPU inside Apple Silicon Macs. **It is NOT CUDA** (CUDA is NVIDIA only). Classic examiner trap. [CODE]
- Every script picks the device in order: MPS → CUDA → CPU. [CODE]
- Your commands use `PYTORCH_ENABLE_MPS_FALLBACK=1`. Meaning: if some operation has no Apple-GPU version, run just that operation on the CPU instead of crashing. It does NOT make everything run on GPU. [PyTorch behavior]
- Verified on your machine: torch 2.14.0, MPS available. [OUTPUT]

---

# 12. CHECKPOINTS (THE .ckpt FILES)

- Each `models/<category>/model.ckpt` is ~72 MB and is actually a **zip file** (PyTorch Lightning checkpoint format). [OUTPUT]
- **Inside:** all network weights (teacher + student + autoencoder), the teacher's normalization statistics, the map-scaling quantiles, the post-processing min/max values, the optimizer state, the epoch number, and the hyperparameters (learning rate, model size).
- **NOT inside:** the dataset, your images, or the threshold from registry.json (that lives in a separate file).
- Loading: `EfficientAd.load_from_checkpoint(path)` rebuilds the model and fills in the weights. [CODE]
- If you changed the architecture (e.g. small → medium), the old checkpoint wouldn't fit — loading would fail.
- They're stored in Git via **Git LFS** (large file storage). Clone without LFS → you get tiny pointer files → the app crashes. The README explains the fix (`git lfs pull`). [CONFIG]

---

# 13. HOW CLOSE IS THIS TO THE ORIGINAL PAPER?

| Aspect | EfficientAD paper | Your project |
|---|---|---|
| Core model (teacher/student/autoencoder) | ✓ | ✓ — via anomalib, faithful [LIB] |
| Training recipe (losses, optimizer, batch 1, 256×256) | ✓ | ✓ — anomalib defaults [LIB] |
| Training length | Long schedule | **Only 20 epochs** — deviation |
| Threshold | Validation-based | **Your own formula on test data** — deviation (weaker) |
| Localization metric | AUPRO | pixel_AUROC instead — partial |
| Speed benchmark | Millisecond latency reported | Not benchmarked by you (measured manually: 39 ms) |
| Deployment | Not covered | Your own web app + registry |

Honest one-liner if asked: *"The model and training are the library's faithful implementation of the paper; my deviations are the 20-epoch budget, my own threshold calibration, and the serving layer."*

---

# 14. BUGS AND WEAKNESSES (BE READY TO ADMIT THESE)

1. **Thresholds tuned on test data** — High (methodology). Leakage → results slightly optimistic. Verbal fix: use a held-out validation set.
2. **Validation data comes from the test split** (anomalib's default behavior) — the normalization stats and internal thresholds also see test data. Same leak, same answer.
3. **`predict_image.py` is broken** — it points to a checkpoint path that doesn't exist. Running it crashes. One-line fix if you need it: change `MODEL_PATH` to `models/metal_nut/model.ckpt`.
4. **The heatmap always shows a red spot somewhere** — because the app stretches colors per image (1st–99th percentile). A perfectly normal image still gets a "hot" area. Colors are relative; only the score decides the verdict. Be ready to explain this if the examiner sees red on a NORMAL result.
5. **`inference_time_seconds` in the metrics is misleadingly named** — it's the wall-clock time for the WHOLE test set, not per-image speed.
6. **20 epochs was not enough for 3 categories** (screw, transistor, capsule — dropped). The instructions file itself suggests 40 epochs for a retry.
7. **Faint scratches get missed** on metal_nut — verified live: scratch scored 0.493 vs threshold 0.5225. [OUTPUT]
8. **No random seeds set** → retraining won't give identical results.
9. **Duplicated code** — the same preprocessing and metrics functions are copy-pasted across 4–5 files.
10. **Thin safety margins** — e.g. tile's threshold (0.501) sits barely above good-image scores (~0.497). Small changes in lighting or camera could flip verdicts.

---

# 15. ANSWERS TO "WHY?" QUESTIONS (QUICK REFERENCE)

- **Why anomaly detection, not classification?** Defects are rare, varied, and unlabeled; normal products are plentiful.
- **Why EfficientAD?** Top-tier accuracy AND very fast (designed for factory-line speed); small enough to run on a Mac.
- **Why MVTec AD?** It's the standard benchmark; has pixel-level ground truth for evaluation.
- **Why teacher–student?** Pretrained teacher knows general images; student learns "normal" from your data; their disagreement on defects is the signal.
- **Why freeze the teacher?** The student needs a fixed target to imitate. Moving target = meaningless comparison.
- **Why normal-only training?** That's the whole point — you can't collect every possible defect, but you can collect good products.
- **Why separate models per category?** Each product's "normal" is completely different.
- **Why 256×256 images?** That's what the EfficientAD architecture is built for (and it keeps things fast).
- **Why batch size 1?** The method's loss is computed per image; anomalib enforces it.
- **Why not YOLO?** Needs thousands of labeled defect boxes; can't handle unseen defect types.
- **Why not a simple autoencoder?** Plain autoencoders often reconstruct defects too well (they generalize) → weak signal. EfficientAD's autoencoder only works in feature space with a tight bottleneck, and it's fused with the teacher–student signal.
- **Why the web app?** Demonstrates actual deployment: model registry, lazy loading (only load a model when first needed — saves ~432 MB of RAM), REST API, simple UI.

---

# 16. THE 10 MINUTE PRE-VIVA MENTAL MODEL

Memorize this flow and you can reconstruct everything:

```text
TRAINING (per category, 6 categories kept):
MVTec train/good images only
→ resize 256×256, batch 1
→ teacher (frozen, pretrained) describes image
→ student learns to copy teacher ON GOOD IMAGES ONLY
→ autoencoder learns global layout
→ loss = imitation error (worst 0.1% patches) + penalty + autoencoder errors
→ 20 epochs, Adam, lr 0.0001, on Apple GPU (MPS)
→ save models/<category>/model.ckpt (72 MB)

CALIBRATION:
run model on test images → threshold = good_max + 0.4×(defect_median − good_max)
→ store in models/registry.json

INFERENCE (web app):
upload photo → RGB, /255, resize 256
→ teacher + student + autoencoder
→ two difference maps → averaged → anomaly map
→ score = maximum value in map (rescaled to ~0.5 range)
→ score ≥ category threshold? DEFECT : NORMAL
→ show colored heatmap
```

And the three numbers to quote from memory: **good nut 0.496, bent nut 0.545, threshold 0.5225** — and the honest failure: **scratch 0.493 = missed**.

---

# 17. STUDY PLAN (2 HOURS)

- **0:00–0:20** — §1, §4 (what the project does + teacher/student/autoencoder). This is 50% of the viva.
- **0:20–0:40** — §5, §6 (training and inference step-by-step). Be able to draw §16 from memory.
- **0:40–0:55** — §7, §8 (score + threshold). Memorize the three real numbers.
- **0:55–1:10** — §9 (metrics) + the results table + the capsule F1 trap.
- **1:10–1:25** — §13, §14 (paper comparison + weaknesses). These disarm the hardest questions.
- **1:25–1:50** — `VIVA_QUESTIONS.md` — read aloud, answer before reading the answer.
- **1:50–2:00** — `VIVA_CHEATSHEET.md` twice, slowly.
