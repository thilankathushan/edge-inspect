# Edge Inspect: implementation plan

**Objective:** Visual anomaly detection and hardware-aware inference.

**Proposed stack:** Python, PyTorch, OpenCV, ONNX Runtime.

**Data:** MVTec AD, initially one category; confirm current license before downloading or publishing.

## Milestone 1: Data and protocol

Create a dataset manifest, license notes, normal-only training split, validation protocol and untouched test set.

**Acceptance gate:** No defective test images influence training, threshold selection or preprocessing.

## Milestone 2: Baseline

Implement a simple image-feature distance baseline, anomaly scores and heatmaps.

**Acceptance gate:** Record image AUROC, defect recall and false-positive rate at a validation-selected threshold.

## Milestone 3: Learned model

Add pretrained feature-based anomaly detection and pixel-level localization evaluation.

**Acceptance gate:** Compare image/pixel metrics against baseline; document failure categories.

## Milestone 4: Edge inference

Export an exportable inference component to ONNX, validate numerical agreement and benchmark warm/cold latency and memory.

**Acceptance gate:** Report hardware, batch size, p50/p95 latency and accuracy before/after optimization.

## Milestone 5: Demo and release

Add image/camera demo, controlled blur/lighting tests and reproducible commands.

**Acceptance gate:** Separate benchmark results from camera-domain tests; publish recorded demo and limitations.

## Portfolio deliverables

README with architecture and setup; reproducible evaluation command; baseline comparison; error analysis; screenshot/demo; short case study describing engineering decisions and limitations. Results are added only after experiments run.

## Completed portfolio scope — 2026-10-25

Milestones 1–5 now have implemented source, a real frozen bottle comparison, normal-only calibration, CPU runtime/export evidence, a local demo, deployment checks and public release documentation. Current evidence and exact verification gates are in docs/RELEASE.md; dated development history remains in PROGRESS.md.

The initial proposal's cold-start benchmarking, controlled blur/lighting study and camera experiments are not part of the verified release. Timings explicitly cover warm inference; no robustness or camera reliability is claimed. These are optional exploratory extensions, alongside quantization, other hardware and public hosting. Any test-informed model improvement requires a separate exploratory record and new held-out data for confirmatory evaluation. Dataset publisher authentication and weight redistribution rights remain limitations that engineering tests cannot resolve.
