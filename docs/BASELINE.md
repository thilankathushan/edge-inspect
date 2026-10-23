# Classical baseline: spatial RGB mean template

Implemented method `rgb-mean-template-v1` follows the baseline fixed in `DATA_PROTOCOL.md`. It needs only NumPy/Pillow and runs on CPU. No pretrained weights, neural network, texture bank, or external model is used.

For each fixed spatial position p, the feature is the three-component RGB vector x(p), in [0,1]. Fit the reference `mu(p) = mean(model_fit RGB vectors)` using float64 accumulation in deterministic manifest order, then store float32. The spatial score is `a(p) = mean_channels((x(p) - mu(p))**2)`; image score is `max_p a(p)`. Scores are in [0,1]. This is a squared Euclidean color-distance baseline with spatial correspondence, not a learned probability. There is no variance normalization, smoothing, learned preprocessing, or calibration contribution to the template.

Preprocessing remains the frozen full-image 256×256 bilinear resize and /255 normalization. Small scratches may disappear on resizing; no texture-specific descriptor is included. Inputs are PNG, JPEG, BMP, or TIFF with recognized extension and decoded format; single-frame RGB or grayscale, sides 2–8192 pixels, at most 32 million pixels. Unsupported modes/alpha/orientation, invalid dimensions, decompression bombs, or corrupt inputs produce actionable errors. Grayscale converts to RGB. Non-square images are stretched as prespecified; predictions on new camera framing may be unreliable.

## Training and model artifact

After official data acquisition and manifest preparation, run from the project root (replace the manifest path with the exact path printed by `prepare-data`):

```sh
.venv/bin/edge-inspect train --config configs/default.toml \
  --manifest "artifacts/manifests/bottle-<sha256>.json" \
  --run-name baseline-v1 --data-kind official_mvtec_ad
```

`--data-kind` is required and is explicitly user-declared provenance, not authentication. Use `synthetic_fixture` for generated data. Runs refuse to overwrite an existing output directory. The manifest category, seed, fixed split allocation, role assignments, and current image hashes are checked. Training reads only `model_fit` and `normal_calibration` images; it neither opens test images/masks nor computes official test performance. Test hashes in manifest metadata are retained solely to prevent inadvertent single-image prediction on reserved data.

`artifacts/models/baseline-v1/` contains:

- `template.npy`: RGB float32 reference, loaded with `allow_pickle=False`.
- `metadata.json`: method/protocol, seed, fit image paths/hashes/count, every normal-calibration score/path/hash, target FPR/rank/threshold/ties/status, manifest digest, dependency versions, implementation source hashes, and data-origin label. The template hash is checked on load, and the saved threshold is recomputed from the saved calibration evidence. These are integrity checks, not a cryptographic signature or provenance proof.

The threshold uses only fixed-model scores from the held-out normal calibration subset. At alpha=0.05, take ascending rank `ceil((n+1)*0.95)` and flag strictly greater scores. Fewer than 19 calibration samples yield infinity internally, JSON `threshold: null`, and `insufficient_calibration`. Prediction then reports `uncalibrated` with `is_anomaly: null`; it does not represent “no flags” as a confident normal classification. The small-sample limitations and exchangeability assumptions are detailed in `DATA_PROTOCOL.md`. No operational recall or measured FPR is established by selecting this threshold.

## Predict a user image

```sh
.venv/bin/edge-inspect predict --config configs/default.toml \
  --model artifacts/models/baseline-v1 \
  --image /absolute/path/to/your-image.png --output-name example-01
```

Results go to `artifacts/predictions/example-01/`:

- `prediction.json`: score, threshold, decision, dimensions, model/input hashes, and model provenance.
- `anomaly_map.npy`: raw float32 256×256 color residual; the maximum of this map is the score.
- `anomaly_map_original.npy`: bilinearly resized residual at the original image dimensions. It is visualization/localization output; its maximum is not used to recompute the image score.
- `overlay.png`: labeled orange residual overlay, explicitly **not defect probability**. Fixed residual range [0,1], square-root display intensity, opacity `0.65*sqrt(residual)`. No per-image min/max stretching or pixel-level threshold. Preview size is bounded to 768 pixels on the longest side; small inputs are enlarged for legibility. Machine-readable maps retain the dimensions above.

`within_threshold` means only that this score did not exceed this model's threshold; it is not a defect-free guarantee. Inputs under the configured official test directory, or byte-identical to reserved test hashes, are rejected. This is an accidental-leakage guard, not protection against deliberately transformed copies or paths from unrelated datasets. `evaluate` remains unimplemented; do not inspect official test performance during development.

## Reproduce fixture verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_fixture_baseline.py --name my-fixture-check
```

The script creates clearly labeled synthetic data, a manifest, an 80-fit/20-calibration model, and a separate synthetic user-image prediction under `artifacts/my-fixture-check/`. Synthetic test-like rows exist only to exercise manifest schema; training never opens them. Choose a new name for each run; existing artifacts are preserved. The script is a software check, not a representative defect demonstration or measured benchmark.

## Expected strengths and limitations

The baseline can respond to localized color changes, missing material, or strongly changed outlines when images are aligned, consistently lit, and similarly framed. Its spatial residual provides a direct explanation of deviations from a normal template.

It is sensitive to translation, rotation, illumination, shadows, reflections, camera changes, and legitimate multimodal appearance. Averaging normal appearances can blur the reference; a defect resembling that average can score low. The maximum score is noise-sensitive. Fine texture defects can be lost at 256×256, and this method has no explicit texture or semantic model. It has no pixel-level decision calibration. Actual defect sensitivity and false-positive behavior are unknown until the frozen final evaluation. No real-data training or calibration has occurred because the official dataset remains unavailable locally.
