# Edge Inspect model card

Release candidate 0.1.0; engineering portfolio demonstration, not a validated production inspector.

## Intended use

Local exploration of one controlled object category, original MVTec AD bottle. Outputs are an image anomaly score, a normal-calibrated operating decision and spatial evidence. Not suitable for safety-critical quality control, arbitrary objects, medical diagnosis, identity inference, or unattended accept/reject decisions. No camera experiments have been performed.

## Models and data

Classical model: spatial mean RGB template and maximum squared residual. Learned model: frozen ImageNet-pretrained ResNet18 layer2, 4096-patch seeded reservoir, exact chunked Euclidean nearest-neighbor scoring. This is PatchCore-style, not the original coreset method. Fixed RGB 256×256 direct resize; learned features use fixed ImageNet normalization. CPU is supported. ONNX exports the backbone only.

Implemented dataset protocol: only official train/good images, deterministically divided into 80% model-fit and 20% normal calibration (rounding documented in DATA_PROTOCOL.md), seed 42. Calibration never enters the template or bank. Threshold uses the fixed 5% target-FPR order-statistic rule and strict greater-than decisions. Fewer than 19 calibration examples cannot provide a finite threshold. Target FPR is not measured deployment FPR.

The real artifacts use 167 bottle normals for fitting and 42 separate normals for calibration. Original MVTec AD bottle was obtained via the pinned third-party mirror in DATA_ACQUISITION.md; transfer integrity is verified, publisher authenticity is not. Baseline threshold 0.3748110830783844; feature threshold 3.768542766571045. Pretrained backbone weights are torchvision ResNet18 IMAGENET1K_V1. Artifact/manifest hashes are in final_evaluation_status.json. Historical fixture artifacts remain clearly labeled.

## Evaluation evidence

The frozen first evaluation used all 83 test images (63 defects, 20 normals). Baseline image AUROC 0.8595, recall 46/63 (73.02%), FPR 1/20 (5%); PatchCore-style AUROC 0.9460, recall 58/63 (92.06%), FPR 2/20 (10%). Macro per-image original-resolution pixel AUROC is 0.8375 versus 0.9679 over 63 eligible anomalous images; 20 normals are undefined/excluded. This is not pooled pixel AUROC. Per-type counts and inspected error analysis are in REAL_RESULTS.md.

Models, preprocessing and thresholds were frozen before test inference. No retuning followed these results. The test split is now exposed; later runtime comparisons are labeled accordingly. The learned model improves overall ranking/recall/localization with more false positives and substantially greater CPU latency. Small samples, shared image conditions and mirror provenance limit external conclusions.

Real-data M2 CPU runtime results and numerical/decision parity are in REAL_ONNX_RESULTS.md. Only the extractor is ONNX; external nearest-neighbor search still requires PyTorch. Historical fixture timings are retained separately in ONNX_RESULTS.md. No camera or other-device result is claimed.


## Limitations and safeguards

Alignment, lighting, pose, background and category shift can dominate scores. Resize may erase tiny defects; features have coarse localization. Maximum scores can be noise-sensitive. The bank may omit normal modes. Heatmaps are not probabilities or pixel-calibrated decisions. Calibration guarantees depend on exchangeability; small samples and correlation limit interpretation. No robustness, fairness, stress/load, security penetration or production monitoring validation has been performed.

Local upload validation, byte/dimension limits and test-hash guards reduce accidental misuse but are not a public-service security boundary. Deploy locally only. Model discovery accepts trusted local artifacts; do not install unknown models. Dataset/weight and code licenses are separate; see ATTRIBUTION.md and RELEASE.md.
