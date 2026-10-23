# CPU PatchCore-style detector

This implementation is **PatchCore-style**, not a reproduction of the original method or its published results. References consulted: Roth et al., [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265), CVPR 2022, and the authors' [official implementation and method guide](https://github.com/amazon-science/patchcore-inspection). The authors' example uses WideResNet50, multiple layers, local patch aggregation and approximate greedy coreset selection. We preserve the core idea of normal pretrained patch embeddings and nearest-neighbor anomaly scores, while making the explicit simplifications below. No code was copied from that repository.

## Frozen configuration and protocol amendment

`configs/frozen_comparison.json` freezes both model configurations before final evaluation. Freeze ID: `bottle-comparison-v2-2026-10-21`. The classical mean-RGB baseline and all data split/preprocessing choices are unchanged. The prior learned-model proposal retained every model-fit patch; this amendment replaces it with a **4096-patch uniform reservoir** to satisfy bounded-memory requirements on the 16 GB CPU machine. No official test performance informed the change. The original `bottle-v1` data manifest remains compatible; its historical `pretrained` description is superseded by the explicit `feature_model` configuration in this comparison freeze and the artifact's `feature_config`.

Settings: torchvision **ResNet-18**, explicit **IMAGENET1K_V1** weights; **layer2 only**, 32×32 grid and 128-dimensional embeddings; CPU, batch size 1, one torch thread, sampling seed 42. Frozen/eval mode disables gradient updates and BatchNorm statistic changes. Extractor graph uses [torchvision create_feature_extractor](https://docs.pytorch.org/vision/stable/feature_extraction.html) to prune downstream layers. No fine-tuning.

Both detectors accept exactly the same raw RGB float32 CHW tensor from the existing full-image 256×256 Pillow bilinear resize and /255 scaling. The feature extractor then applies fixed ImageNet mean/std once, internally. It does not crop, use model-fit normalization statistics, or apply normalization twice. The optional `preprocess_image(pretrained=True)` route is tested for numerical agreement, but callers of `.score()` must pass the raw [0,1] tensor.

Each layer2 vector represents a receptive-field patch. No explicit local pooling, multi-layer concatenation, learned projection, greedy/farthest-point coreset, scoring reweighting, Gaussian heatmap smoothing, or model ensemble is implemented. A smaller single-layer backbone and uniform sampling trade representativeness for a simple bounded implementation. Unlike full PatchCore coreset selection, rare normal modes can be missed.

## Bank, scoring, and memory limits

Only `model_fit` rows enter the bank. Process images in deterministic manifest order and patches in row-major order. Algorithm R reservoir sampling, using NumPy PCG64 seed 42, retains at most 4096 embeddings uniformly over the streamed patch population. Record each retained patch's fit-image path and patch index. Calibration is processed only after the bank is complete and cannot modify it. Test images/masks are never opened by training.

The bank payload is at most `4096 × 128 × 4 = 2,097,152` bytes (2 MiB), excluding Python provenance metadata and allocator overhead. Each image contributes a temporary 1024×128 float32 feature array (0.5 MiB). No full training embedding matrix is accumulated. Exact Euclidean 1-NN scoring uses query chunks of 64 and bank chunks of 512, with direct `torch.cdist`; each returned distance block is at most 32,768 floats (128 KiB). These are allocation bounds for those arrays, **not measured total process RSS**. Backbone weights, activations, loaded manifests, metadata, framework runtime, and transient deserialization allocations add overhead.

Image score is the maximum of the raw 32×32 patch-distance map. The spatial heatmap is bilinearly upsampled to 256×256 with `align_corners=False`; original-size visualization follows the shared prediction path. Interpolation can reduce the maximum, so the image score is not recomputed from the resized map. Distances are nonnegative, unbounded, and not probabilities.

Use the same held-out-normal rank threshold as the baseline: target FPR 5%, `k=ceil((n+1)*0.95)`, flag strictly `score > threshold`. Fewer than 19 calibration images yields an infinite threshold represented as JSON null and an uncalibrated decision. Finite thresholds do not establish real FPR or recall. The overlay clips display values at the saved finite threshold; if unavailable, at the maximum saved calibration score, floored at 1e-6. It uses a square-root orange intensity, explicitly labels this scale, and never normalizes by the query image. That image-level scale is not a pixel-level threshold.

## Dependencies and explicit weight download

The optional extra pins torch 2.13.0 / torchvision 0.28.0, a pair listed for Python through 3.14 in [the official compatibility table](https://github.com/pytorch/vision#installation). The installed macOS arm64 wheels support CPU. `certifi` supplies a project-local CA bundle for verified HTTPS; no system certificate changes or TLS-verification bypass is used. `requirements-features.lock.txt` records the resolved feature environment; `requirements-dev.lock.txt` remains the lightweight baseline environment.

```sh
.venv/bin/python -m pip install -e '.[dev,features]'
# Explicit network action: ~45 MB, stored under ignored artifacts/weights.
.venv/bin/edge-inspect fetch-weights --config configs/default.toml
```

The sole download URL is `https://download.pytorch.org/models/resnet18-f37072fd.pth`, published in [torchvision's ResNet source](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html). The [weight definition](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) identifies IMAGENET1K_V1. Download writes a temporary file, verifies the published SHA-256 prefix `f37072fd`, then installs it and writes a source receipt. Training/load/prediction never download; missing weights fail with an explicit fetch instruction. Model construction always uses `weights=None` before loading the verified local state.

Observed local download on 2026-10-21: 46,830,571 bytes; SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Weights and saved extractor states stay outside Git.

### Weight license status

Verified [torchvision's BSD-3-Clause code license](https://github.com/pytorch/vision/blob/main/LICENSE). This is **not evidence that the weight file has identical unrestricted terms**. [Torchvision's official pretrained-weight notice](https://docs.pytorch.org/vision/stable/models.html) says weights may have their own licenses or terms derived from training datasets. No separate ResNet18 IMAGENET1K_V1 weight-specific license was identified on the official weight definition/source. [ImageNet's dataset access terms](https://www.image-net.org/download.php) limit the database to non-commercial research/educational use; those database terms are not asserted here to be a complete weight license.

The precise weight-specific license therefore remains unverified. The artifact records that limitation rather than inventing a permissive license. This local portfolio research implementation does not assert commercial deployment or redistribution rights. Do not bundle weights in a public release without resolving the intended use and applicable permissions. MVTec's separate dataset terms still apply to real training inputs.

## Train and predict

After preparing a real-data manifest, substitute its exact path:

```sh
.venv/bin/edge-inspect train-features --config configs/default.toml \
  --manifest "artifacts/manifests/bottle-<sha256>.json" \
  --weights artifacts/weights/resnet18-f37072fd.pth \
  --run-name patchcore-style-v1 --data-kind official_mvtec_ad
.venv/bin/edge-inspect predict --config configs/default.toml \
  --model artifacts/models/patchcore-style-v1 \
  --image /absolute/path/to/user-image.png --output-name feature-example-01
```

Prediction detects the artifact method and uses the appropriate model. Existing outputs are never overwritten. Artifacts contain `memory_bank.npy`, pruned `extractor.pth`, and JSON metadata with full preprocessing/backbone/layer/sampling/chunk configuration, source weight URL/hash/license status, bank patch provenance, fit records, calibration evidence, source-code hashes, dependency versions, manifest digest, and test-reservation hashes. Loading uses `allow_pickle=False` for NumPy and `weights_only=True` for torch, validates file hashes and frozen configuration, and recomputes the stored threshold from saved calibration evidence. Hashes are integrity checks, not authenticity signatures. Stored extractor weights make offline inference portable without the original download cache.

## Tests and verification limits

```sh
.venv/bin/python -m pytest -q
```

Feature tests need the optional extra; actual-backbone integration tests need the explicitly fetched weight file. They skip with a reason when unavailable and never fetch during tests. Tests cover reservoir determinism/bounds/origins, chunked search versus brute force, preprocessing agreement, eval/frozen CPU mode, fit-only bank provenance, absence of test-file reads, save/load map/score parity, calibration changes leaving the bank unchanged, prediction geometry, and rejected configuration changes.

Local checks use official pretrained weights with **synthetic software fixtures**, not official MVTec training/calibration or test data. No defect metrics, latency benchmark, or total-memory benchmark is claimed. Real models and thresholds must be fitted and their artifact digests frozen before the prespecified final comparison. `evaluate` remains unimplemented.
