# VIVA QUESTIONS — project-specific, with answers

Format: **Q** / short answer (say aloud) / detailed answer / likely follow-up → follow-up answer.

---

## BASIC

**Q1. What does your project do?**
Short: Detects industrial defects as anomalies — models trained only on normal products flag anything unusual, with a score, heatmap, and NORMAL/DEFECT verdict in a web app.
Detailed: One EfficientAD model per MVTec AD category, trained on `train/good` only, served via Flask. Six categories deployed.
Follow-up: Why not a classifier? → No labeled defect data at scale; defects are rare, diverse, and unseen types occur.

**Q2. What kind of ML problem is this?**
Short: One-class / unsupervised anomaly detection with image-level scoring and pixel-level localization.
Detailed: `LearningType.ONE_CLASS` in anomalib; training data contains zero defects.
Follow-up: Is it fully unsupervised? → Semi-supervised in the loose sense: normal class is curated; evaluation is supervised (labels+masks) but training never sees them.

**Q3. Walk me through the pipeline end to end.**
Short: MVTec → resize 256, batch 1 → train teacher-frozen student+AE 20 epochs → checkpoint per category → calibrate threshold → Flask app: preprocess → teacher/student/AE → fused anomaly map → max score → threshold → verdict + heatmap.
Follow-up: Where's the decision? → `app.py` line 247: `is_defect = score >= threshold`.

**Q4. What did YOU build vs what does the library do?**
Short: Anomalib implements the EfficientAD model/training; I built the multi-category training pipeline, threshold calibration, model registry with tiers, the web app, and the webcam calibration/voting layer.
Detailed: `src/` has no network definitions; model internals are in `.venv/.../anomalib/models/image/efficient_ad/`.
Follow-up: So you didn't implement the paper? → Correct — I integrated and validated a faithful library implementation, and I can explain its internals (see Architecture section).

**Q5. Which dataset and why?**
Short: MVTec AD — the standard industrial AD benchmark: real textures, small defects, pixel ground truth, normal-only training split.
Follow-up: What's in train vs test? → train = only `good/`; test = `good/` + several defect types; `ground_truth/` has pixel masks for evaluation only.

---

## ARCHITECTURE

**Q6. Describe EfficientAD's components.**
Short: Frozen pretrained teacher PDN (384 ch), student PDN with double channels (768), and an autoencoder; anomaly = where student fails to imitate.
Detailed: PDN = 4 convs + 2 avg-pools, receptive field ~33×33, output 64×64 for 256² input. AE has 64-d bottleneck → captures global/logical structure.
Follow-up: Why two student halves? → channels 0–383 imitate the teacher (local/textural anomalies), channels 384–767 imitate the AE (logical anomalies).

**Q7. Why is the teacher frozen?**
Short: It defines a stable reference feature space; the student learns relative to it.
Detailed: Teacher weights are pretrained (distilled from EfficientNet on ImageNet, distributed by anomalib). It runs under `no_grad`, in `eval()`, and Adam receives only student+AE parameters (`configure_optimizers`).
Follow-up: Does the teacher adapt to your data at all? → Only its channel **mean/std statistics** are computed on the training set and used to normalize its features.

**Q8. What does the student actually learn?**
Short: To reproduce the teacher's features for normal patches — and nothing else.
Detailed: Trained only on normal images with a hard-example loss, plus a penalty pushing its output toward zero on ImageNet images so it doesn't generalize its imitation beyond the normal domain.
Follow-up: Why does it fail on anomalies? → No training signal for anomalous patches; its weights are specialized to the normal distribution, so its output there is essentially wrong → large squared distance.

**Q9. What is the autoencoder for?**
Short: Catching logical defects — misplaced/missing parts — that patch-local comparison misses.
Detailed: AE compresses to a 64-d global vector, so it reconstructs teacher features only when the global arrangement is normal. Student's second half imitates the AE; disagreement flags logical anomalies.
Follow-up: What happens if removed? → map_stae disappears; textural detection remains, logical detection degrades (transistor 'misplaced' type cases).

**Q10. Why is this fast ("Efficient")?**
Short: Tiny fully-convolutional networks, one forward pass, shallow 64×64 feature grid — millisecond latency by design.
Detailed: PDN-S is ~4 conv layers; no backbone-sized ResNet at inference. Paper reports millisecond CPU latency.
Follow-up: What's your actual latency? → Measured 39 ms/image mean on MPS (model + post-processing only). `inference_time_seconds` in metrics JSON is whole-test-set wall time — not latency.

**Q11. What are the tensor shapes through the model?**
Short: 1×3×256×256 → teacher 1×384×64×64, student 1×768×64×64, AE 1×384×64×64 → maps 1×1×64×64 → upsampled 1×1×256×256 → scalar score.
Follow-up: Why 64×64? → two stride-2 avg-pools: 256→128→64.

---

## EFFICIENTAD THEORY

**Q12. Explain knowledge distillation here.**
Short: Student trained to match teacher's feature maps on normal data; distillation failure on anomalies is the detection signal.
Detailed: Loss = squared feature distance on the hardest 0.1% of locations; not class logits but intermediate CNN features.

**Q13. Why the hard-example loss (99.9% quantile)?**
Short: Focus learning on the student's worst errors instead of easy background patches.
Detailed: `d_hard = quantile(d_st, 0.999)`; loss = mean of distances above it. Prevents the loss being dominated by trivially-easy regions.

**Q14. What's the ImageNet penalty for?**
Short: Keep the student "blank" off-distribution so anomalies stay detectable.
Detailed: `loss_penalty = mean(student(ImageNette_image)²)`; without it the student could learn to imitate the teacher on arbitrary images, shrinking the teacher–student gap on defects.

**Q15. Why does the AE see an augmented image?**
Short: Brightness/contrast/saturation jitter makes AE features robust to appearance variation that isn't structural.
Detailed: `choose_random_aug_image` applies one random photometric transform with coefficient ~U(0.8,1.2).

---

## PYTORCH / CODE

**Q16. Where's the forward pass in YOUR code?**
Short: `app.py` line 125: `model.post_processor(model.model(tensor))`.
Detailed: `model` is the Lightning module; `.model` is the inner `EfficientAdModel`. In eval it returns `InferenceBatch(pred_score, anomaly_map)`.
Follow-up: Why `post_processor`? → Min-max normalizes the score using validation min/max stored in the checkpoint.

**Q17. Why `@torch.inference_mode()`?**
Short: Disables autograd tracking — faster, less memory, guarantees no gradient graph.
Follow-up: vs `no_grad`? → Stricter/faster; tensors can't later require grad.

**Q18. Why batch size 1 in training?**
Short: Paper requirement; the hard-loss quantile is computed per image, and anomalib raises an error otherwise (`on_train_start` check).
Follow-up: Could you use batch 8? → You'd have to change anomalib internals; quantile semantics change.

**Q19. Why no `Normalize` transform in preprocessing?**
Short: The model applies ImageNet normalization internally (`imagenet_norm_batch` inside the PDN); anomalib raises if the pre-processor contains Normalize.
Follow-up: What does your preprocessing do then? → BGR→RGB, HWC→CHW, /255, resize 256.

**Q20. What's in the .ckpt file?**
Short: A Lightning checkpoint (zip): state_dict (teacher+student+AE weights, mean/std, quantiles, post-processor min/max + thresholds), optimizer state, epoch, hyperparameters.
Detailed: ~72 MB each, stored via Git LFS.
Follow-up: Can you resume training? → Format supports it, but my scripts don't pass `ckpt_path` to `fit` — not wired.

**Q21. Why lazy model loading in app.py?**
Short: Six 72 MB models; loading all eagerly wastes ~432 MB RAM and slows startup.
Detailed: `MODEL_CACHE` dict + `threading.Lock` because Flask is multi-threaded — prevents double-loading races.

**Q22. Why does `train_all.py` skip existing checkpoints?**
Short: Idempotent re-runs — retrain only missing categories; `--force` overrides.
Follow-up: Risk of `--force`? → It deletes the checkpoint before training; a crash loses the old model. Better: train to temp then swap.

---

## DATASET

**Q23. Why one model per category?**
Short: "Normal" for a bottle and a cable are disjoint distributions; separate models are more accurate and give per-category thresholds.
Follow-up: Cost? → storage, user must select category, no cross-category generalization.

**Q24. What happens if I give the metal_nut model a cable image?**
Short: Garbage — likely a huge anomaly score and meaningless heatmap; it's fully out-of-distribution.
Follow-up: How does the app prevent this? → User selects the category; the app uses that category's model and threshold.

**Q25. What if a defect type is unseen in testing too?**
Short: That's the setting the method is built for — anything deviating from learned normal can fire; risk is only if the deviation is too subtle (faint scratch on metal_nut scored 0.493 < 0.5225 threshold — a real miss I verified).

**Q26. What if you have very few normal training images?**
Short: Teacher mean/std stats and map quantiles get noisy; student underfits; thresholds unstable.

**Q27. Why does MVTec have ground-truth masks if training never uses them?**
Short: For pixel-level evaluation — pixel_AUROC and pixel_F1 measure localization quality against those masks.

---

## TRAINING

**Q28. Trace `engine.fit` for one category.**
Short: on_train_start (load pretrained teacher, download ImageNette, compute teacher mean/std) → per-step three-loss training on student+AE → on_validation_start computes map quantiles → checkpoint.
Follow-up: What does Adam optimize? → `student.parameters() + ae.parameters()` only.

**Q29. What's the learning rate schedule?**
Short: Adam lr 1e-4, weight decay 1e-5, StepLR decays ×0.1 at 95% of total steps.
Detailed: `configure_optimizers` in anomalib's lightning_model.py.

**Q30. How many epochs, and is that enough?**
Short: 20 for every category. Honest answer: enough for bottle/tile/leather/metal_nut (AUROC ≥0.97), not for transistor/screw/capsule — those were dropped; instructions note retraining with 40 epochs.
Follow-up: Why fixed 20? → Time constraint on MPS; per-category tuning is future work.

**Q31. Where do checkpoints and metrics land?**
Short: Lightning logs under `results/`; best checkpoint copied to `models/<category>/model.ckpt`; metrics to `metrics/<category>.json`.
Follow-up: What's "best" here? → `engine.best_model_path` from Lightning's checkpoint callback.

**Q32. What are the three training losses?**
Short: Student–teacher hard distillation (+ImageNet penalty), AE imitation of teacher, student-AE imitation — summed unweighted.
Detailed: See Q13/Q14 math.

---

## INFERENCE

**Q33. Exactly how is the anomaly score computed?**
Short: Fused map = 0.5·(teacher–student map) + 0.5·(AE–student map), quantile-normalized, upsampled; score = map maximum; then min-max normalized to validation range.
Detailed: `torch.amax(anomaly_map, dim=(-2,-1))` in anomalib's forward.
Follow-up: Why max and not mean? → Defects are small/local; mean would dilute them below detectability. Cost: sensitivity to single-pixel noise.

**Q34. Is the score a probability?**
Short: No — a normalized squared feature distance. 0.55 is not 55% defective; it's only comparable to that category's threshold.
Follow-up: Is it calibrated? → Min-max to validation range, not probabilistically calibrated.

**Q35. Why do all my scores sit around 0.5?**
Short: Anomalib's post-processor min-max normalizes using validation statistics, mapping the normal operating range near the middle of [0,1].

**Q36. Why can two visually similar images get different scores?**
Short: Score is a max over 65k pixels — a single hot pixel (specularity, noise, slight pose change) shifts it; also per-model normalization differences across categories.

**Q37. How does the anomaly map localize defects?**
Short: The PDN is fully convolutional — each 64×64 feature cell corresponds to a ~33×33 input patch; upsampling the cell values back to 256² gives a spatial heatmap.
Follow-up: Why is localization imperfect? → 4× downsampling + bilinear upsample blurs small defects; wood pixel_F1 is only 0.52.

**Q38. Why does a normal image still show red regions in your heatmap?**
Short: The web app stretches each map by its own 1st–99th percentiles for contrast — colors are relative per image, not absolute. Verdict comes only from score vs threshold. That's a visualization tradeoff I acknowledge.

---

## MATHEMATICS

**Q39. Write the student–teacher distance.**
Short: `d_st[c,i,j] = (T_norm[c,i,j] − S[c,i,j])²`, then channel-mean → `map_st[i,j]`.
Detailed: T_norm uses channel mean/std computed on the training set.

**Q40. Write the total loss.**
Short: L = (mean of d_st above its 99.9% quantile + mean S(ImageNet)²) + mean(T−AE)² + mean(AE − S[384:])².
Follow-up: Any weighting? → No, plain sum.

**Q41. How are the maps normalized before fusion?**
Short: `map' = 0.1·(map − qa)/(qb − qa)` with qa=90%, qb=99.5% quantiles computed on validation good images; then 0.5/0.5 average.

**Q42. Why squared distance and not L1 or cosine?**
Short: Squared L2 penalizes large deviations strongly, matching the "student fails hard on anomalies" signal; it's also the paper's choice and keeps gradients simple.

**Q43. Define image_AUROC.**
Short: Probability that a randomly chosen defective image scores higher than a randomly chosen good image; threshold-free.
Follow-up: Why not accuracy? → Needs a threshold; AUROC compares models independent of operating point — that's why it drove model selection.

**Q44. Define F1 and why it can mislead.**
Short: 2PR/(P+R) at a fixed threshold. Capsule had F1 0.911 with AUROC 0.688 — predicting "defect" almost always gives high recall; without AUROC you'd be fooled.

---

## METRICS & RESULTS

**Q45. Which metrics do you compute?**
Short: image_AUROC, image_F1, pixel_AUROC, pixel_F1 (via anomalib's engine.test), plus whole-test-set wall time. No AUPRO, no per-image latency.
Follow-up: Why no PRO? → Not implemented; pixel_AUROC is my localization proxy — honest gap.

**Q46. Best and worst deployed category and why?**
Short: bottle 1.000 AUROC (large rigid object, big obvious defects); cable 0.928 (thin structures, many subtle defect types), hence tier-2 "experimental".

**Q47. Your pixel_F1 is much lower than image_F1 — why?**
Short: Detection ≠ precise segmentation: the score needs one hot pixel; F1 over pixels needs the whole defect boundary correct. Upsampled 64×64 maps blur edges.

**Q48. How were the 6 categories chosen?**
Short: image_AUROC ≥ 0.95 → tier-1 (bottle, tile, leather, metal_nut); 0.90–0.95 → tier-2 (wood, cable); below → dropped (screw 0.862, transistor 0.778, capsule 0.688).

**Q49. Could your metrics be optimistic?**
Short: Yes, two reasons: anomalib's validation comes from the test split (quantiles, min-max, adaptive threshold see test data), and my deployed thresholds are tuned on the test set. I'd fix it with a proper held-out validation split.
(This answer turns your biggest weakness into your strongest moment. Use it.)

---

## DEBUGGING

**Q50. Normal images being flagged as defects — how do you debug?**
Short: Check threshold vs good-score distribution (recalibrate), check preprocessing (BGR/RGB!), check lighting/pose drift, check heatmap location (edges/specularities), check quantization stats loaded from ckpt.

**Q51. A defect scores NORMAL — debug?**
Short: Look at the map: is the defect visible but below max elsewhere (localization fine, threshold issue) or invisible (feature blind spot)? metal_nut faint scratch = real example: 0.493 vs 0.5225.

**Q52. Model won't load — causes?**
Short: Git LFS pointer file instead of real ckpt (README covers `git lfs pull`); anomalib version mismatch; architecture change (state_dict shape mismatch).

**Q53. AUROC high but webcam predictions bad — why?**
Short: Domain shift — webcam images differ from MVTec in lighting/sensor/pose; that's why realtime.py refuses the dataset threshold and calibrates in-situ with a known-good object.

**Q54. Training loss not decreasing?**
Short: Check teacher weights loaded, mean/std computed, lr too high, batch≠1 error, MPS fallback warnings hiding CPU thrash.

---

## DESIGN DECISIONS

**Q55. Why not YOLO?**
Short: Needs thousands of labeled defect boxes per defect type; can't detect unseen defect types. Wrong tool for rare-diverse-defect QC.

**Q56. Why not a ResNet binary classifier?**
Short: Same label problem + closed-world assumption: a classifier learns "defects I've seen", anomalies are open-set.

**Q57. Why not pure autoencoder reconstruction?**
Short: Classic AEs generalize too well — they reconstruct anomalies too, shrinking the error signal. EfficientAD's AE only imitates teacher *features* with a tight bottleneck and is fused with the distillation branch.

**Q58. Why Flask and not just a script?**
Short: Demonstrates deployment: registry-based multi-model serving, lazy loading, REST API usable by other systems, UI for non-technical inspection.

**Q59. Why is threshold 0.4·gap and not ROC-optimal?**
Short: Honest answer: heuristic chosen for a quick separation guarantee; the right method is validation ROC/Youden or target false-positive-rate selection — identified as future work.

**Q60. Why MPS / what is MPS fallback?**
Short: MPS = PyTorch's Apple Silicon GPU backend. `PYTORCH_ENABLE_MPS_FALLBACK=1` lets ops without Metal kernels run on CPU instead of crashing — not a CUDA thing, and it can silently slow those ops.

---

## LIMITATIONS

**Q61. Biggest technical weakness?**
Short: Test-set leakage in threshold calibration and validation-derived normalization; 20-epoch training budget; faint-scratch misses; per-image heatmap stretching can mislead visually.

**Q62. Is this production-ready?**
Short: It's a demonstration. README §10 says so: real deployment needs on-site normal data, matched camera geometry/lighting, retrained/adapted models, validated thresholds.

**Q63. Reproducibility?**
Short: Weak — no seeds set; training runs vary; checkpoints + registry + anomalib 2.6.2 pin are the reproducibility anchors.

**Q64. Single point of failure in the scoring?**
Short: Max-based score — one noisy pixel can flip a verdict; mitigated in realtime.py by temporal voting (5 of 7 frames) but not in the web app.

---

## PAPER COMPARISON

**Q65. Did you implement EfficientAD exactly as the paper?**
Short: The model/training core is Anomalib's faithful implementation (same PDN teacher/student/AE, hard loss, ImageNet penalty, 256², batch 1, Adam 1e-4). My deviations: 20-epoch budget vs paper's longer schedule, my own threshold calibration instead of paper/anomalib validation thresholding, no PRO metric, and my own serving layer.

**Q66. What does the paper evaluate that you don't?**
Short: AUPRO (localization) and millisecond latency benchmarks — I have neither; my timing is whole-test-set wall clock.

---

## REAL-TIME

**Q67. Is your system real-time?**
Short: No. The product is an upload-based web app. The model itself is fast — I measured 39 ms per image on MPS, a ~25 FPS model-only ceiling — but the webcam script is a legacy unmaintained demo with frame skipping and 5-of-7 voting lag, so it is not a real-time deployment.
Follow-up: What would real-time need? → drop/loosen voting, process every frame, async inference pipeline, benchmark end-to-end latency not just model time.

**Q67b. Explain the legacy webcam pipeline.**
Short: 640×480 frame → 300×300 ROI crop → same preprocessing → score every N frames (`--skip`) → in-situ calibration threshold (mean+3σ over 30 frames of a known-good object) → 5-of-7 temporal vote → overlay verdict + heatmap.
Follow-up: Why voting? → kills single-frame false alarms at the cost of confirmation lag.

**Q68. Why does realtime.py refuse the dataset threshold?**
Short: Domain shift — webcam score distribution differs from MVTec's; the code explicitly sets label to NORMAL until calibrated ("We don't trust the default MVTec threshold for webcam").

**Q69. What limits webcam FPS?**
Short: Model forward is 39 ms/image (measured) — not the main limiter. The limiters are the Python/OpenCV capture loop, frame skipping (`--skip N`), and the 5-of-7 voting window that delays any DEFECT verdict by several inferences.

---

## HARDWARE

**Q70. What device runs training/inference?**
Short: Apple Silicon via MPS (`mps`→`cuda`→`cpu` fallback chain in `get_device`). Verified torch 2.14, MPS available.

**Q71. MPS vs CUDA?**
Short: MPS = Apple Metal backend; CUDA = NVIDIA. Code is written device-agnostic through PyTorch.

---

## EXAMINER ATTACK SET

**Q72. "How does the model know what normal looks like?"**
Through the student: it sees every normal training image 20 times and is optimized to make teacher−student distance small on exactly that distribution, plus teacher channel statistics and validation quantiles anchor the scale.

**Q73. "Show me where thresholding happens."**
`app.py` line 247 (`is_defect = score >= threshold`), threshold from `models/registry.json` written by `src/calibrate_thresholds.py` (formula at lines 90–95). Real-time version: `realtime.py` line 322.

**Q74. "What does this dimension represent?"** (pointing at 1×768×64×64)
Batch 1; 768 = 2×384 student channels (teacher-imitating half + AE-imitating half); 64×64 = spatial grid after two stride-2 pools, each cell ≈ 33×33 px receptive field.

**Q75. "What happens if I change the threshold?"**
Lower → recall up, false positives up; higher → precision up, missed defects. Current values sit just above the good-score max (e.g. tile 0.501 vs good ≈0.497) — thin margin, so even small drift flips verdicts; that's a robustness limitation I can defend.

**Q76. "How do you know you're not overfitting?"**
Training data has no defects, so classic overfit-to-labels is impossible; but the student can overfit the normal set (memorize) — mitigated by hard-loss mining, AE bottleneck, ImageNet penalty. Evidence of generalization: test good images score in the same band as calibration (metal_nut good 0.496 vs threshold 0.5225).

**Q77. "Why should I trust this model?"**
Bounded trust: per-category image_AUROC 0.93–1.00 on a standard benchmark, verified live behavior (bent nut detected, scratch missed), explicit tier labels, documented failure modes, and known methodological leaks (test-set calibration) that I'd fix before production.

**Q78. "What happens if I remove the autoencoder?"**
Lose map_stae (half the fused map); logical/misplacement defects degrade; textural detection mostly intact.

**Q79. "Why separate models instead of one big model?"**
Disjoint normal manifolds per category; a joint model broadens "normal" until defects hide inside it; separate models also give clean per-category thresholds at the cost of storage and a category selector.

**Q80. "What's the receptive field and why does it matter?"**
~33×33 px for PDN-S. It sets the minimum defect scale the teacher–student comparison can see; defects smaller than that blur into their neighborhood — part of why faint scratches and screw's tiny defects underperform.
