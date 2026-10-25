# Frozen evaluation protocol and current status

**2026-10-25: real frozen evaluation complete.** See [results and error analysis](REAL_RESULTS.md), [pre-run record](REAL_EVALUATION_PROTOCOL.md), and [mirror provenance](DATA_ACQUISITION.md). The test split is now exposed; subsequent runs are exploratory.

## Before opening held-out results

`configs/frozen_comparison.json` fixes the baseline and PatchCore-style settings. Train both on the identical manifested model-fit subset; calibrate only on normal calibration images using the prespecified 5% target-FPR rule. Do not change these choices based on official test results.

`evaluate` checks model provenance, manifest identity, fit/calibration membership, frozen configurations, runtime library versions, source hashes, and test inventory. It loads artifacts, then writes `prerun_snapshot.json` and a manifest copy **before test inference**. The snapshot includes model files' SHA-256 hashes, full metadata and calibration thresholds/scores, protocol, hardware, runtime versions and evaluator source hashes. Evaluation never trains or recalibrates.

Official attempts are registered in `artifacts/evaluations/heldout-ledger`. Reusing the same test identity requires `--exploratory`. Failed attempts retain their ledger entry and partial logs; do not erase the ledger to claim a fresh holdout. This local guard is not tamper-proof: moving the artifact root or changing the dataset can bypass it. Keep the full evaluation history and disclose any prior inspection. User-declared dataset provenance is not independent publisher authentication.

## Metrics

- Image AUROC uses exact ranking, positive = defect, half credit for ties. Both classes are required.
- Decisions use strict `score > saved_threshold`; confusion counts, defect recall TP/(TP+FN), FPR FP/(FP+TN), and class/sample counts are recorded. The calibration target does not guarantee observed test FPR. An unavailable finite calibration threshold produces null decisions/counts/rates with reasons.
- Pixel AUROC is computed exactly **within each image at its original mask resolution**, then averaged equally across images containing both foreground and background pixels. The 256×256 float anomaly map is resized bilinearly; original binary masks are unchanged. Normal-only and full-foreground masks have undefined within-image AUROC and are excluded with counts/reasons. This is **macro per-image AUROC, not globally pooled pixel AUROC**; do not compare it directly with published pooled results. Positive/negative pixel totals include excluded images.
- Per-defect results compare each type against all shared normal test images. AUROC requires at least two examples in each group; counts and rates are retained wherever defined. Groups smaller than five defects are flagged. These are descriptive results without confidence-interval claims.
- JSON uses null plus reasons for undefined metrics, never NaN or misleading zeroes. Known-answer tests cover ranking/ties, strict thresholds, absent classes and aggregation.

## Latency boundaries

CPU, RGB 256×256, batch size one, one PyTorch thread. Five warmup calls use the first model-fit normal, never a selected test example. Each test image is measured once in manifest path order, baseline first then features. `perf_counter_ns` records score-only time and validation/decode/resize plus score time. Score-only includes feature normalization, nearest-neighbor search and map interpolation. Hash checks, model loading, masks, original-resolution map resizing, metrics, plotting and output writes are excluded. Decoder validation is included in pipeline timing. File caches are warm/uncontrolled; results are not cold-start latency. Report p50/p95/mean and retain every measurement. Hardware/OS/library versions are recorded. This is not a total-memory benchmark; separate process-RSS measurements are reported in REAL_ONNX_RESULTS.md.

## Outputs and error analysis

Each unused run directory contains the pre-run snapshot, manifest, status, `results.json`, per-model `predictions.jsonl`, raw maps, `comparison.png`, `scores.png`, and `REPORT.md`. One median-score example per TP/FN/FP/TN (or uncalibrated) bucket is chosen deterministically, with path tie-breaking. Empty buckets are explicitly absent. Overlays show original input, original mask and labeled anomaly map. Display scales use the baseline's fixed residual range or feature calibration threshold, never test-derived optimization.

Reports describe observed errors and localization diagnostics (within-image AUROC, region means, whether the maximum lies in the mask). Possible illumination/alignment/resize causes are clearly hypotheses; the report cannot establish causality. The completed real error analysis is in REAL_RESULTS.md. Later improvements must be separately labeled exploratory.

Metric definition reference: [official AUROC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html); implementation here uses NumPy and exact tie-aware ranks, without a scikit-learn dependency. Plotting uses [Matplotlib](https://matplotlib.org/stable/users/installing/index.html).
