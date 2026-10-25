# Real bottle evaluation — pre-run record (2026-10-25)

Dataset: original MVTec AD bottle obtained from the third-party Hugging Face mirror, revision c75b39616f84db43677bcc8228caaafaf5096d7f. Publisher authenticity has not been independently established. Integrity validation passed. This is real bottle imagery, not generated fixtures.

The configuration freeze remains `configs/frozen_comparison.json`. No test inference or visual test inspection informed the settings below. Tests/masks were decoded only for integrity checks. Preprocessing, backbone, layer, sampling, score and threshold rules are unchanged from the fixture-development freeze.

- Manifest SHA-256: e1bbf78c4b1ecd2dbc5f0394877dfc02c753efb31ea5fcdd8b358154e60ff546.
- Model-fit: 167 normals. Calibration: 42 other normals. Test: 83 images, 20 normal and 63 defective, with original masks.
- Classical baseline: mean RGB template, mean-channel squared residual, maximum image score, RGB 256×256 direct bilinear resize /255.
- Learned model: ResNet18 IMAGENET1K_V1 layer2 (128×32×32), 4096-patch seeded reservoir, exact chunked Euclidean 1-NN; maximum raw patch distance; bilinear map resize.
- Seed 42; normal-only 5% target-FPR rule, rank 41 of 42, strict score > threshold.
- Baseline threshold: 0.3748110830783844. Feature threshold: 3.768542766571045.
- Saved models: artifacts/models/bottle-baseline-v1 and artifacts/models/bottle-patchcore-v1. Actual artifact digests, metadata and acquisition provenance will be recorded by the evaluator before test inference.
- Evaluation definitions: docs/EVALUATION.md, including exact image AUROC and macro within-image original-resolution pixel AUROC (not pooled).
- Timing: actual M2 CPU, batch one, 256×256, one torch thread, five fit-normal warmups. Initial evaluation measures one call per test; subsequent runtime comparison will repeat calls in separate processes.

The first frozen test run will be retained even if the learned model underperforms. No threshold/model changes will be made after seeing test results. ONNX checks on the same test images are explicitly a later runtime parity comparison on exposed data, not a new untouched evaluation. No quantization or camera experiments are included.
