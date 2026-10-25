# ONNX runtime results — 2026-10-23

Historical synthetic-fixture results. Current real bottle measurements: [REAL_ONNX_RESULTS.md](REAL_ONNX_RESULTS.md).

**Actual Apple M2 measurements on synthetic software fixtures only. Official MVTec bottle data is still absent.** These establish software parity and observed runtime behavior, not real defect-detection performance.

Only the ResNet18 layer2 extractor was exported. Preprocessing, normalization, patch layout, torch nearest-neighbor scoring, map interpolation and the unchanged calibration threshold remain external. Fixed FP32 batch 1, 256×256, opset 18, CPUExecutionProvider, one intra/inter-op thread, sequential ORT with full graph optimization and spinning disabled. No quantization.

Hardware: Apple M2, 8 CPU cores, 16 GB unified memory; macOS 27.0.1 arm64; Python 3.14.0. PyTorch 2.13.0 / torchvision 0.28.0, ONNX 1.23.2 / onnxscript 0.7.2 / ONNX Runtime 1.31.0, psutil 7.2.2.

## Measured latency

Five full warmups on a model-fit normal. Calibration run: 20 normal fixtures × 5 repetitions = 100 observations per runtime. Exposed-fixture run: 3 normal + 3 anomaly fixtures × 10 repetitions = 60 observations per runtime. Each backend runs in a fresh process, original first then ONNX; tests were not run concurrently. End-to-end spans validation/decode through score and 256×256 map, excluding hashing, model loading, masks, metrics and output/overlay writing.

| Workload / component | Original p50 / p95 ms | ONNX path p50 / p95 ms |
|---|---:|---:|
| onnx-normal-calibration / preprocessing | 0.737 / 0.860 | 0.721 / 0.828 |
| onnx-normal-calibration / embedding | 13.804 / 14.883 | 30.629 / 31.002 |
| onnx-normal-calibration / scoring | 255.538 / 260.372 | 255.910 / 257.854 |
| onnx-normal-calibration / end_to_end | 270.128 / 275.477 | 287.216 / 289.549 |
| onnx-exposed-fixtures / preprocessing | 0.727 / 0.861 | 0.724 / 0.777 |
| onnx-exposed-fixtures / embedding | 13.825 / 14.592 | 30.689 / 31.122 |
| onnx-exposed-fixtures / scoring | 255.646 / 257.884 | 255.984 / 261.246 |
| onnx-exposed-fixtures / end_to_end | 270.163 / 272.511 | 287.397 / 293.782 |

**ONNX was slower in both measured workloads**, about 6.3–6.4% at median end-to-end latency. Its embedding step took about 31 ms versus 14 ms; the unchanged nearest-neighbor scorer dominated at about 256 ms. No settings were tuned after these measurements. Keep the original runtime as the default. These repeated warm measurements have fixed backend order, uncontrolled thermal/cache/system conditions and no confidence intervals; do not generalize them to other hardware or data.

## Parity and quality

Fixed embedding/score/map tolerances were atol=rtol=1e-4, specified before measurement. All passed. Exact threshold decisions changed on **0/20 calibration fixtures and 0/6 exposed fixtures**. No thresholds or memory-bank entries were changed.
- onnx-normal-calibration: maximum embedding absolute error 5.9604645e-06; maximum image-score error 1.9073486e-06.
- onnx-exposed-fixtures: maximum embedding absolute error 8.8810921e-06; maximum image-score error 1.4305115e-06.

Calibration AUROC/defect recall are undefined because there are no anomalies; both runtimes made zero positive decisions on those 20 calibration fixtures. This in-sample normal result does not establish future FPR.

For the six **already-evaluated synthetic fixtures**, both runtimes produced identical observed metrics: image AUROC 0.777778, recall 0.666667, FPR 0.333333, TP/FN/FP/TN = 2/1/1/2, macro within-image pixel AUROC 0.954487 (3 eligible anomaly images, 3 excluded normal images). This comparison is explicitly exploratory; it neither resets the test holdout nor establishes real detector quality. The original frozen fixture results remain unchanged.

## Process memory

| Workload | Original lifetime peak RSS MiB | ONNX path lifetime peak RSS MiB |
|---|---:|---:|
| onnx-normal-calibration | 377.5 | 349.4 |
| onnx-exposed-fixtures | 378.8 | 349.7 |

ONNX showed lower whole-worker peak RSS in these runs. This is **not model-only memory**: fresh-process `resource.ru_maxrss` includes interpreter/framework imports, load/cache allocations, parity-array output and pixel-metric work. psutil stage RSS before/after load, warmup and measurement is available in JSON and each full report. Shared resident pages and allocator retention affect the figures. No GPU/system memory or Raspberry Pi/other-device benchmark was performed.

## Reproduction and retained evidence

[Export and benchmark commands/methodology](ONNX.md) describe model paths, boundary, threading, parity rules and limitations. `requirements-onnx.lock.txt` records the environment. The actual runs use `artifacts/fixture-evaluation-verified/config.toml`, its single hashed manifest, `outputs/models/features`, and `outputs/exports/resnet18-layer2-fp32`. Use an unused run name. Add `--evaluation artifacts/fixture-evaluation-verified/outputs/evaluations/comparison` only for the exposed-fixture run.

Full ignored artifacts:
- `artifacts/fixture-evaluation-verified/outputs/benchmarks/onnx-normal-calibration/`: snapshot, job/input identities, JSON results, raw repeated timing samples, per-image embeddings/maps, memory values and REPORT.md.
- `artifacts/fixture-evaluation-verified/outputs/benchmarks/onnx-exposed-fixtures/`: snapshot, job/input identities, JSON results, raw repeated timing samples, per-image embeddings/maps, memory values and REPORT.md.

[Compact machine-readable results](onnx_results.json) retain the measured summaries and snapshot hashes without weights or dataset files. Original frozen evaluation outputs were preserved; these runtime reports are separate. Real-data quality/parity and deployment performance remain pending official acquisition and real training. Quantization is deferred; any later experiment needs separate normal-only calibration and quality reporting.
