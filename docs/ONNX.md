# ONNX extractor and runtime benchmark

## Export boundary

The existing detector uses a frozen torchvision ResNet18 through layer2. Its fixed convolution/batch-normalization/ReLU/residual/pooling graph exports to ONNX. **This is only a feature extractor, not a complete ONNX anomaly detector.**

The graph takes `normalized_rgb`: FP32 NCHW **[1,3,256,256]**, already normalized using the saved ImageNet mean/std. It outputs `features`: FP32 **[1,128,32,32]**. Opset **18**, fixed shapes, batch one, FP32, no external tensor file. The command uses the PyTorch dynamo exporter and ONNX checker, then performs a seeded synthetic-input embedding check. Export loads the saved source state; it does not fetch weights or train anything. Bitwise-identical ONNX files across exporter versions are not promised; versions and file hashes are recorded.

Outside ONNX: Pillow validation/decode/RGB/bilinear resize; division by 255; NumPy mean/std normalization; row-major 1024×128 patch layout; the existing bounded exact torch.cdist nearest-neighbor search; torch bilinear map interpolation; maximum-distance image score; strict comparison to the saved normal-calibration threshold. The bank, threshold, backbone state and frozen configuration are unchanged. The classical baseline is unchanged and is not exported.

The runtime validates the ONNX hash and its binding to the original model metadata/state. Prediction accepts `--onnx` for a feature model and retains the existing test-input guards. It still requires PyTorch for external scoring. No full-pipeline ONNX portability or minimal deployment footprint is claimed.

## Commands

Install optional dependencies in the project environment:

```sh
.venv/bin/python -m pip install -e '.[dev,features,evaluation,onnx]'
```

For an existing model (substitute paths to the same model/config/manifest throughout):

```sh
.venv/bin/edge-inspect export-onnx --config configs/default.toml \
  --model artifacts/models/patchcore-style-v1 --run-name resnet18-layer2-fp32
.venv/bin/edge-inspect predict --config configs/default.toml \
  --model artifacts/models/patchcore-style-v1 \
  --onnx artifacts/exports/resnet18-layer2-fp32 \
  --image /absolute/path/to/user-image.png --output-name onnx-user-01
.venv/bin/edge-inspect benchmark-onnx --config configs/default.toml \
  --manifest "artifacts/manifests/bottle-<sha256>.json" \
  --model artifacts/models/patchcore-style-v1 \
  --onnx artifacts/exports/resnet18-layer2-fp32 \
  --run-name onnx-calibration --warmup 5 --repeats 5
```

By default, benchmark inputs are **only normal calibration images**. This tests speed/parity and normal operating decisions without opening held-out test images. It cannot establish defect recall or AUROC. It does not change the threshold, recalibrate, train, or rebuild the bank.

To compare quality after a completed frozen evaluation, add `--evaluation artifacts/evaluations/frozen-bottle-v1` and choose a new run name. The command requires matching model/manifest hashes and a completed report. It uses that already-exposed test split and labels the run exploratory. It never presents this as a fresh held-out evaluation. The image and original-resolution macro pixel metrics use the existing frozen evaluation definitions. No model setting is selected from these results.

## Parity and timing protocol

Before inference, the benchmark records the source model metadata/hashes, calibration threshold, export specification/hash, runtime versions, code hashes, input identities, hardware, warmup/repetition counts and protocol. Fixed tolerances are `atol=rtol=1e-4` for embeddings, image scores and maps. Final decisions are checked exactly with strict `score > original_threshold`; a numerically close score can still flip a decision at a tie. Such changes are reported, never hidden through retuning. No finite threshold means uncalibrated decisions, not normal predictions.

Each runtime runs in a fresh sequential subprocess, original first then ONNX. CPU only, batch one, FP32, 256×256, PyTorch intra/inter-op threads one; ORT intra/inter-op threads one, sequential execution, `CPUExecutionProvider`, full graph optimization, spinning disabled. BLAS/OpenMP thread environment variables are set to one. No CPU affinity, thermal control, or exclusive machine access is promised.

Five full score/map warmups use the first model-fit normal. Every selected input is measured repeatedly (default five passes). `perf_counter_ns` records preprocessing, embedding extraction, external scoring/map interpolation, and their contiguous end-to-end duration. Component timing includes Python/API overhead. End-to-end here means validated image file through score and 256×256 map; it excludes model load, integrity hashing, masks, metrics, serialization, original-resolution rendering and overlays. Hashing warms file caches; this is not a cold-start benchmark. Samples are correlated; p50/p95 describe this workload, not a statistical confidence interval or a universal speedup.

## Memory methodology

Each worker records psutil process RSS before model load, after load, after warmup and after the measurement loop. `resource.ru_maxrss` records its process-lifetime high-water mark, converted from bytes on macOS or KiB on Linux. Measurements include Python/framework imports, model loading, retained allocator/cache memory, parity-array saving and pixel-metric work. Both paths still import PyTorch/torchvision for scoring. High-water RSS is not model-only allocation; stage samples can miss transient peaks. Shared resident pages are counted, and OS accounting/allocator behavior affects comparisons. No GPU allocation or system-wide RAM measurement is claimed.

## Outputs, current evidence and limits

Each ignored `artifacts/benchmarks/<run-name>` includes the pre-run snapshot/job, separate worker logs/results, per-image embeddings/maps, raw per-repeat timing/score samples, aggregate JSON, parity details and a Markdown report. Preserve these with the exact source artifacts. Exported files remain outside Git and inherit the unresolved pretrained-weight licensing limitations documented in PATCHCORE_STYLE.md.

Real bottle training, frozen evaluation and runtime comparison are complete; see [real-data runtime results](REAL_ONNX_RESULTS.md). [Earlier synthetic-fixture measurements](ONNX_RESULTS.md) remain historical software checks. All runs use the actual M2 CPU; none establish Raspberry Pi, GPU or other-device performance.

Quantization is deferred. It changes embeddings relative to the retained FP32 memory bank and may change calibration decisions; it needs a separate, normal-only calibration protocol and separately disclosed quality evaluation. There is no assumed benefit from adding it now.

Official references used: [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx), [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), [ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [psutil RSS documentation](https://psutil.readthedocs.io/stable/). Installed versions are recorded in the benchmark snapshot and `requirements-onnx.lock.txt`.
