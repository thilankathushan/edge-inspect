# Edge Inspect model card

Release candidate 0.1.0; engineering portfolio demonstration, not a validated production inspector.

## Intended use

Local exploration of one controlled object category, original MVTec AD bottle. Outputs are an image anomaly score, a normal-calibrated operating decision and spatial evidence. Not suitable for safety-critical quality control, arbitrary objects, medical diagnosis, identity inference, or unattended accept/reject decisions. No camera experiments have been performed.

## Models and data

Classical model: spatial mean RGB template and maximum squared residual. Learned model: frozen ImageNet-pretrained ResNet18 layer2, 4096-patch seeded reservoir, exact chunked Euclidean nearest-neighbor scoring. This is PatchCore-style, not the original coreset method. Fixed RGB 256×256 direct resize; learned features use fixed ImageNet normalization. CPU is supported. ONNX exports the backbone only.

Intended dataset protocol: only official train/good images, deterministically divided into 80% model-fit and 20% normal calibration (rounding documented in DATA_PROTOCOL.md), seed 42. Calibration never enters the template or bank. Threshold uses the fixed 5% target-FPR order-statistic rule and strict greater-than decisions. Fewer than 19 calibration examples cannot provide a finite threshold. Target FPR is not measured deployment FPR.

**Current artifacts are trained on synthetic software fixtures. Official bottle data has not been acquired, so real training and held-out evaluation are absent.** Pretrained backbone weights are real downloaded torchvision weights; fixture predictions are actual inference, not fabricated model output.

## Evaluation evidence

All official test images remain reserved. The evaluator records manifest/model hashes and thresholds before inference, reports image AUROC, confusion counts, recall/FPR, per-defect results and macro within-image pixel AUROC at original mask resolution. The latter is not pooled pixel AUROC. Undefined metrics include reasons. Subsequent test-informed runs are exploratory.

Fixture evaluations test software behavior only. Apple M2 runtime measurements found the ONNX extractor path slower end-to-end (calibration-fixture p50 287.2 vs 270.1 ms), with lower whole-process peak RSS (349.4 vs 377.5 MiB). No decisions changed across 26 checked fixtures. See ONNX_RESULTS.md for sample counts, p95, versions and boundaries. No claim of real defect accuracy, edge-device performance or statistical superiority follows.

## Limitations and safeguards

Alignment, lighting, pose, background and category shift can dominate scores. Resize may erase tiny defects; features have coarse localization. Maximum scores can be noise-sensitive. The bank may omit normal modes. Heatmaps are not probabilities or pixel-calibrated decisions. Calibration guarantees depend on exchangeability; small samples and correlation limit interpretation. No robustness, fairness, stress/load, security penetration or production monitoring validation has been performed.

Local upload validation, byte/dimension limits and test-hash guards reduce accidental misuse but are not a public-service security boundary. Deploy locally only. Model discovery accepts trusted local artifacts; do not install unknown models. Dataset/weight and code licenses are separate; see ATTRIBUTION.md and RELEASE.md.
