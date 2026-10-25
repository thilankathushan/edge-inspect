# Portfolio release status — 2026-10-25

**Published source repository:** [github.com/thilankathushan/edge-inspect](https://github.com/thilankathushan/edge-inspect), public `main`. Source license: MIT. Code, tests, aggregate plots, documentation and permitted UI assets are published; datasets and model artifacts are excluded. The demo is local, not hosted publicly. This release makes no production-readiness claim.

## Completed

- [x] Configurable package, validator, hashed manifest, deterministic model-fit / normal-calibration split and held-out guard.
- [x] Frozen RGB-template baseline and bounded ResNet18 PatchCore-style model trained on 167 fit-only normals. Both use 42 separate calibration normals; thresholds saved before test inference.
- [x] First frozen test evaluation on 83 images: metrics, per-defect results, local overlays and inspected error analysis in REAL_RESULTS.md and ignored artifact outputs.
- [x] ONNX feature extractor export and real calibration/test parity and speed comparisons; original remains faster.
- [x] Local Streamlit app exercised with saved real baseline and learned models, plus a clearly labeled generated sample. Upload validation, loading guidance, score, threshold, heatmap, overlay and timing work.
- [x] Real model provenance appears in UI. Public screenshots/GIF use synthetic geometry only; no dataset images or artifacts are released.
- [x] Python 3.11/3.14 lightweight CI workflow, no required dataset or model downloads. Remote workflow previously passed. The new offline demo tests pass locally and will run again when this update is pushed.
- [x] 103 full local tests passed; 88 CI-selected lightweight tests passed. `pip check` and release audit passed. The audit found no Git-visible credentials, home paths, >5 MiB files, model/data archives or unreviewed images; heuristic scanning cannot prove that no secret exists.
- [x] Baseline and CPU feature Docker images built for arm64 Linux. Direct no-network model smoke checks passed with actual read-only-mounted artifacts, non-root UID 10001 and rejected invalid uploads. Image contents contain neither dataset nor model files.
- [x] MIT code license decision. Dataset, pretrained weights and dependencies retain separate terms.

## Evidence and limits

The MVTec bottle bytes came from the pinned third-party Hugging Face mirror because the publisher download failed. Hashes verify transfer relative to that mirror; publisher authenticity has not been independently established. Dataset files and notices remain local. The dataset license is CC BY-NC-SA 4.0; attribution, non-commercial and ShareAlike terms apply. Review them before reuse or redistribution.

Pretrained ResNet18 weights come from torchvision. A separate weight-specific redistribution/commercial license was not verified. No weight files, trained detector states, banks or ONNX exports are in Git. This does not prevent running the source locally, but anyone redistributing weights or derived artifacts needs to resolve the applicable terms.

The evaluation uses one category and modest samples. The 42-image calibration sample produced a 5% target-FPR threshold, but test FPR was 5% for the baseline and 10% for PatchCore-style (one and two false alarms among only 20 normal tests). The test set has been viewed and must not be presented as untouched again. Test-informed improvements need a distinct new test set for confirmatory claims.

ONNX only contains the feature extractor and was slower end-to-end. RSS is whole-worker process memory; timing is warm, single-thread CPU. No camera, quantization, MPS, Raspberry Pi, other edge hardware, production security, robustness, load, or deployment-service tests are claimed. The captured GIF is a silent sequence of actual UI states, not a continuous narrated recording.

Both images built and actual model inference smoke checks passed. Docker Desktop later became manually paused, so the HTTP health endpoint and Compose lifecycle remain unchecked after the build. Resume Docker Desktop to run `docker compose up -d` and `curl -fsS http://127.0.0.1:8501/_stcore/health` locally. Native Streamlit browser use passed on loopback port 8503.

## Optional owner actions

- Record narration following DEMO_SCRIPT.md if a continuous video is useful for your portfolio.
- Resolve dataset mirror authenticity and model-weight rights before any use that depends on publisher provenance, commercial use or redistribution.
- Resume Docker Desktop and run the health command above if you plan to feature the container path.

There is no required local action to view the source, read the results, or start the native demo with the author's ignored local artifacts present. A fresh clone does not include models; use the labeled fixture quickstart or acquire permitted data, fetch weights explicitly, and train locally.
