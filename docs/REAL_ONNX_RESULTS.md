# Real bottle CPU runtime comparison — 2026-10-25

These measurements use the frozen real bottle model. Data provenance: [third-party mirror](DATA_ACQUISITION.md). The first quality evaluation is [reported separately](REAL_RESULTS.md). No model/threshold changes or quantization followed the test results. Historical fixture measurements remain in ONNX_RESULTS.md.

Normal calibration: 42 inputs × 5 repetitions (210 timings per backend). Already-exposed test split: 83 inputs × 3 repetitions (249 timings per backend), explicitly a later runtime/parity comparison. Benchmarks ran sequentially without concurrent project tests or Docker builds. Machine-wide exclusivity was not enforced.

**ONNX was slower on both workloads. Numerical checks passed and there were zero changed decisions across 42 calibration and 83 test images.** The original runtime remains the default. These are warm file-to-score/map measurements, not app response time or cold-start measurements.

[Machine-readable summary](real_onnx_results.json).

## Normal calibration

Mode: `normal_calibration_runtime_check`.

Observed on the recorded local hardware only.

Hardware: {'os': 'macOS-27.0.1-arm64-arm-64bit-Mach-O', 'architecture': 'arm64', 'python': '3.14.0', 'logical_cpu_count': 8, 'cpu': 'Apple M2', 'memory': '16 GB', 'device': 'cpu', 'cpu_description': 'proc 8:0:4:4'}
Versions: {'numpy': '2.5.4', 'Pillow': '13.0.0', 'torch': '2.13.0', 'torchvision': '0.28.0', 'onnx': '1.23.2', 'onnxscript': '0.7.2', 'onnxruntime': '1.31.0', 'psutil': '7.2.2'}

Only ResNet18 through layer2 is ONNX. Preprocessing/normalization, patch layout, torch nearest-neighbor scoring, map resize and the unchanged threshold remain external. No quantization.

Warmup: 5 full calls on a fit normal; repeats: 5 per input; batch 1, 256x256 FP32, one intra/inter-op thread, CPUExecutionProvider, sequential ORT, full graph optimization, spinning disabled.

| Component | Original p50 / p95 ms | ONNX path p50 / p95 ms |
|---|---:|---:|
| preprocessing | 21.903 / 23.364 | 21.934 / 23.428 |
| embedding | 12.828 / 13.942 | 30.290 / 31.259 |
| scoring | 256.108 / 260.328 | 258.941 / 266.928 |
| end_to_end | 291.272 / 295.594 | 311.132 / 320.937 |

Observed median end-to-end ratio original/ONNX: 0.936x; ONNX path was slower or equal in this run. No confidence interval or universal speedup claim.

Timings exclude load, file hashing, masks, metrics and output writing. Embedding includes normalization/layout; scoring includes map resize. End-to-end means validated file decode through score/map, without serialization/overlay rendering. Caches are warm, backend order fixed, and repeated inputs are correlated.

### Numerical parity and quality

Numerical parity passed: True. Fixed atol=rtol=1e-4 for embeddings, score and maps. Decision changes: **0 / 42** at the unchanged calibration threshold.
Maximum embedding absolute error: 1.090765e-05; score absolute error: 3.5762787e-06.

| Runtime | Image AUROC | Recall | FPR | TP/FN/FP/TN | Pixel macro AUROC |
|---|---:|---:|---:|---|---:|
| original | None | None | 0.023809523809523808 | 0/0/1/41 | None |
| onnx | None | None | 0.023809523809523808 | 0/0/1/41 | None |

None means undefined; JSON contains reasons and counts. Calibration-only inputs cannot establish defect recall or AUROC. Pixel aggregation matches the frozen original-resolution macro protocol. Test-mode inputs were already evaluated: this is an exploratory runtime comparison, not a fresh holdout. No threshold is adjusted to hide changed decisions.

### Process memory

Fresh process per runtime. psutil current RSS before model load, after load, after warmup and after measurements. resource ru_maxrss process-lifetime high-water mark (macOS bytes, Linux KiB converted). Includes Python/framework imports, source validation, load, caches, parity array saving and pixel-metric work; not incremental model allocation, GPU memory or whole-system RAM. Stage RSS can miss transient peaks; high-water includes startup. Shared pages counted; allocator retention and OS effects apply.

| Runtime | Before load MiB | After load MiB | After warmup MiB | After run MiB | Lifetime peak MiB |
|---|---:|---:|---:|---:|---:|
| original | 202.1 | 346.5 | 381.3 | 383.2 | 392.2 |
| onnx | 201.6 | 323.7 | 346.7 | 361.0 | 375.0 |

Both paths still import PyTorch for scoring. ONNX is not a standalone lightweight full detector. No MPS, Raspberry Pi, or other edge hardware was benchmarked. Quantization is deferred: this FP32 comparison must be understood before a separately calibrated/evaluated tradeoff.


## Previously exposed test split

Mode: `exploratory_runtime_comparison_of_previously_evaluated_test`.

Observed on the recorded local hardware only.

Hardware: {'os': 'macOS-27.0.1-arm64-arm-64bit-Mach-O', 'architecture': 'arm64', 'python': '3.14.0', 'logical_cpu_count': 8, 'cpu': 'Apple M2', 'memory': '16 GB', 'device': 'cpu', 'cpu_description': 'proc 8:0:4:4'}
Versions: {'numpy': '2.5.4', 'Pillow': '13.0.0', 'torch': '2.13.0', 'torchvision': '0.28.0', 'onnx': '1.23.2', 'onnxscript': '0.7.2', 'onnxruntime': '1.31.0', 'psutil': '7.2.2'}

Only ResNet18 through layer2 is ONNX. Preprocessing/normalization, patch layout, torch nearest-neighbor scoring, map resize and the unchanged threshold remain external. No quantization.

Warmup: 5 full calls on a fit normal; repeats: 3 per input; batch 1, 256x256 FP32, one intra/inter-op thread, CPUExecutionProvider, sequential ORT, full graph optimization, spinning disabled.

| Component | Original p50 / p95 ms | ONNX path p50 / p95 ms |
|---|---:|---:|
| preprocessing | 22.269 / 24.288 | 22.333 / 23.939 |
| embedding | 13.490 / 15.492 | 30.710 / 33.563 |
| scoring | 260.299 / 288.358 | 262.076 / 285.383 |
| end_to_end | 295.904 / 329.006 | 315.163 / 343.648 |

Observed median end-to-end ratio original/ONNX: 0.939x; ONNX path was slower or equal in this run. No confidence interval or universal speedup claim.

Timings exclude load, file hashing, masks, metrics and output writing. Embedding includes normalization/layout; scoring includes map resize. End-to-end means validated file decode through score/map, without serialization/overlay rendering. Caches are warm, backend order fixed, and repeated inputs are correlated.

### Numerical parity and quality

Numerical parity passed: True. Fixed atol=rtol=1e-4 for embeddings, score and maps. Decision changes: **0 / 83** at the unchanged calibration threshold.
Maximum embedding absolute error: 1.0311604e-05; score absolute error: 2.8610229e-06.

| Runtime | Image AUROC | Recall | FPR | TP/FN/FP/TN | Pixel macro AUROC |
|---|---:|---:|---:|---|---:|
| original | 0.946031746031746 | 0.9206349206349206 | 0.1 | 58/5/2/18 | 0.9678996514323583 |
| onnx | 0.946031746031746 | 0.9206349206349206 | 0.1 | 58/5/2/18 | 0.9678996489711033 |

None means undefined; JSON contains reasons and counts. Calibration-only inputs cannot establish defect recall or AUROC. Pixel aggregation matches the frozen original-resolution macro protocol. Test-mode inputs were already evaluated: this is an exploratory runtime comparison, not a fresh holdout. No threshold is adjusted to hide changed decisions.

### Process memory

Fresh process per runtime. psutil current RSS before model load, after load, after warmup and after measurements. resource ru_maxrss process-lifetime high-water mark (macOS bytes, Linux KiB converted). Includes Python/framework imports, source validation, load, caches, parity array saving and pixel-metric work; not incremental model allocation, GPU memory or whole-system RAM. Stage RSS can miss transient peaks; high-water includes startup. Shared pages counted; allocator retention and OS effects apply.

| Runtime | Before load MiB | After load MiB | After warmup MiB | After run MiB | Lifetime peak MiB |
|---|---:|---:|---:|---:|---:|
| original | 201.8 | 346.7 | 382.0 | 389.5 | 436.9 |
| onnx | 201.7 | 323.7 | 346.8 | 388.1 | 422.5 |

Both paths still import PyTorch for scoring. ONNX is not a standalone lightweight full detector. No MPS, Raspberry Pi, or other edge hardware was benchmarked. Quantization is deferred: this FP32 comparison must be understood before a separately calibrated/evaluated tradeoff.
