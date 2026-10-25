# Edge Inspect progress

## 2026-10-21 — Initial inspection

- Read the workspace `ROADMAP.md` and this project's `README.md` and `IMPLEMENTATION_PLAN.md`. No applicable `AGENTS.md` was found in the project, workspace, or ancestor directories.
- Verified that the project initially contained only `README.md` and `IMPLEMENTATION_PLAN.md`. No code, datasets, models, demo, or evaluation outputs exist yet.
- Inspected Git layout with `git rev-parse --show-toplevel`, `git status --short`, and a search for `.git` entries. This project belongs to the workspace repository; Project 1 has its own nested repository. Workspace content is currently untracked in the workspace repository.
- Added this progress log. No implementation tests or benchmarks were run; no performance or quality results are available.

## Working constraints

- Work only inside `02-edge-inspection`; do not modify other projects.
- Implement one prompt at a time, starting with a reproducible baseline before model complexity.
- Detect defects in one object category, produce scores and heatmaps, compare a simple baseline with pretrained features, and provide a local demo and reproducible evaluation.
- Confirm dataset licensing before downloading or publishing. Keep test data isolated from training, preprocessing decisions, and threshold selection.
- Measure inference latency and memory on actual hardware. Clearly distinguish synthetic demonstrations from real dataset benchmarks.
- Do not commit or push until publication is requested. Future commits must use Thilanka Thushan <158076487+thilankathushan@users.noreply.github.com>, without AI co-author trailers.
- Update this log with implemented features, commands, verification results, limitations, and the next step.

## 2026-10-21 — Foundation implemented

### Features

- Added an installable `src/edge_inspect` package, `pyproject.toml`, project-local `.venv`, and resolved development dependency snapshot in `requirements-dev.lock.txt`.
- Installed Pillow 13.0.0 and pytest 9.1.1 with their small dependencies. No system-wide installation, dataset, model framework, or model-weight download occurred.
- Added strict TOML configuration for paths, category, seed, artifacts, and CPU execution. Paths resolve relative to the config; unknown keys, invalid types/ranges, unsafe category names, and overlapping directories fail clearly.
- Added `check-config`, read-only `inspect-data`, JSON stderr logging, and honest nonzero placeholders for `train`, `evaluate`, and `predict`.
- Inspection checks decoding, normal-only training folders, image layout, and symlinks, and reports per-split/label counts. Tiny synthetic test images are software fixtures only.
- Added project-local ignore rules and updated the README with setup, commands, architecture boundaries, official dependency references, and limitations.

### Hardware observations

- `python3 --version`: Python 3.14.0, using the Python.framework installation.
- `sw_vers` / `uname -m`: macOS 27.0.1, arm64.
- `system_profiler SPHardwareDataType`: MacBook Air, Apple M2, 8 CPU cores (4 performance + 4 efficiency), 16 GB memory.
- `system_profiler SPDisplaysDataType`: Apple M2 GPU, 8 cores, Metal supported.
- `sysctl` hardware query was denied by the sandbox; System Profiler supplied the needed hardware information. No device identifiers are recorded here.
- CPU remains the only enabled execution device. MPS is a hardware possibility, not verified framework acceleration; PyTorch is deliberately deferred to the model milestone.

### Commands and verification

Run from this project folder:

- `python3 -m venv .venv`: succeeded.
- `.venv/bin/python -m pip install --no-cache-dir -e '.[dev]'`: succeeded after retrying with network access; the initial sandbox attempt could not resolve the package index.
- `.venv/bin/python -m pytest -q`: **26 passed**. Covers config-relative paths, invalid configuration, missing/empty data, corrupt images, invalid layout, defective training folders, symlinks, JSON output/logs, and explicit placeholder failures.
- `.venv/bin/python -m pip check`: no broken requirements.
- `.venv/bin/python -m pip freeze --exclude-editable > requirements-dev.lock.txt`: recorded resolved dependencies without a machine-specific editable path.
- `.venv/bin/edge-inspect check-config --config configs/default.toml`: succeeded with resolved project-local paths and CPU default.
- `.venv/bin/python -m edge_inspect --help`: succeeded; all five commands are discoverable.
- `git check-ignore .venv/probe data/probe models/probe artifacts/probe sample.onnx`: all excluded.
- No commits, pushes, Git identity changes, or edits to other projects.

### Limitations and next step

Only Python 3.14.0 on this Mac has been tested; the declared Python range is 3.11–3.14. The dependency snapshot pins versions but is not a hash-locked cross-platform environment. No anomaly detector, scores, heatmaps, dataset split protocol, mask-pair validation, evaluation, inference benchmark, or demo exists yet. Seeds are validated configuration for future randomized operations, not a claim of model determinism. `bottle` is provisional.

Next command: `.venv/bin/edge-inspect check-config --config configs/default.toml`.

Next implementation milestone: confirm the dataset/category and license, then implement the manifest and isolated training/validation/test protocol. No data download is authorized by this foundation task.

## 2026-10-21 — Bottle data pipeline

### Implemented

- Selected **bottle from original MVTec AD**, as requested. Prespecified protocol and initial model choices in `docs/DATA_PROTOCOL.md` before any real test evaluation.
- Extended read-only inspection with image mode/orientation validation, complete image decoding, mask pairing and original dimensions, binary nonempty masks, missing/orphan-mask errors, duplicate-image hash checks, and corrupt-file handling. No invalid samples are silently dropped.
- Added `prepare-data`: deterministic JSON manifests with SHA-256 for every image and paired mask, relative paths, category, official split/labels, anomaly flags, dimensions, library versions, protocol and role. Snapshots use their SHA-256 in the filename and atomic replacement after successful validation.
- Added seeded hash-order splitting of official normal training only: `ceil(0.20*N)` calibration images and the rest fit images. Official test rows retain `final_test`. Manifest preparation checks that test normals and masked anomalies are present, but does not prove official dataset completeness.
- Added deterministic 256×256 bilinear RGB float32 CHW preprocessing, /255 scaling, optional fixed ImageNet normalization, and nearest-neighbor boolean masks. No learned statistics or random transformations.
- Added streaming role-based loading with raw-file hash checks, protocol consistency checks, split/path consistency checks, and explicit final-test opt-in. Raw files are not rewritten or moved. Software guards do not replace adherence to the documented experimental protocol.
- Added/tested a normal-only calibration threshold helper: 5% target FPR, rank `ceil((n+1)*0.95)`, strict score > threshold, infinity when no finite rank is available. Documented exchangeability assumptions, finite-sample uncertainty, and the prohibition on tuning with calibration or test anomalies.
- NumPy 2.5.4 installed only inside `.venv`; `requirements-dev.lock.txt` refreshed. Installation approach checked against https://numpy.org/install/.

### Official source and acquisition status

- Verified https://www.mvtec.com/research-teaching/datasets/mvtec-ad and the linked CC BY-NC-SA 4.0 license/deed/legal code on 2026-10-21. Commercial use is excluded; sharing/adaptations carry attribution and other license conditions. Documented source/license links, modification notices, attribution template, and original paper reference.
- Inspected the official download form: required first name, last name, email, and acknowledgment of non-commercial use; newsletter is optional. Did not submit personal information or acknowledge terms.
- **Acquisition pending:** category-only download support, current archive URL/size, and further terms are not visible until the form is completed. No guessed legacy URL, third-party mirror, or full archive used. The user must supply the official bottle download link after completing the form, or provide the contact details and explicit authorization needed to continue that process. Explain any full-archive size/terms requirement before downloading.
- No real dataset has been downloaded, visually inspected, used to tune choices, or evaluated. Structural test-data validation in the implementation is integrity-only. No benchmark results exist.

### Verification

- `.venv/bin/python -m pip install --no-cache-dir -e '.[dev]'`: succeeded in project-local environment.
- `.venv/bin/python -m pytest -q`: **58 passed**. Fixtures are tiny generated images/masks, explicitly labeled as synthetic software tests. Checks include hash correctness, byte-identical repeated manifests, raw-file preservation, disjoint fit/calibration roles, seed behavior, independence from test-content changes, invalid/missing/orphan masks, input corruption, duplicate content, missing test classes, test-loading rejection, tampered split assignments, changed image/mask hashes, changed protocol rejection, exact normalization and mask geometry, threshold ranks/ties/invalid scores, and CLI output.
- `.venv/bin/python -m pip check`: no broken requirements.
- `.venv/bin/python -m pip freeze --exclude-editable > requirements-dev.lock.txt`: dependency snapshot refreshed.
- `.venv/bin/edge-inspect check-config --config configs/default.toml` and `.venv/bin/python -m edge_inspect --help`: succeeded.
- No commits, pushes, or modifications outside Project 2.

### Next step and limitations

Complete the official acquisition flow and confirm a bottle-only download. Once raw data is in `data/mvtec_ad/bottle`, run `.venv/bin/edge-inspect prepare-data --config configs/default.toml`. Until then that command reports missing data rather than creating a synthetic replacement. The real schema, content hashes, counts, and archive provenance remain unverified.

The pipeline cannot detect near-duplicate/semantic leakage or authenticate a download using locally computed hashes alone. It rejects exact duplicate images conservatively. Initial model choices are documented, but models, inference, actual thresholds, evaluation, benchmarks, and the demo remain unimplemented. Test samples must remain reserved for the frozen final comparison.

## 2026-10-21 — End-to-end classical baseline

### Implemented

- Implemented the already prespecified `rgb-mean-template-v1` classical spatial color method. Per-position RGB vectors are compared to a mean template fitted **only** on model_fit normals. Float64 accumulation yields a stored float32 template; mean squared channel distance gives the 256×256 spatial map, and its maximum gives the image score. No deep-learning dependency, learned normalization, variance fitting, or model-choice change was introduced.
- `train --manifest ... --run-name ... --data-kind ...` checks seed/category, fixed split assignments, normal roles, hashes, and duplicate/leakage conditions. It streams fit and calibration samples separately. Official test images/masks are never opened during training; reserved test hashes are metadata only.
- Saved `template.npy` (no pickle) plus JSON metadata: template/manifest/source hashes, package versions, seed, input provenance, fit records, normal-calibration scores and hashes, finite-sample rank, target FPR, threshold, tie count, and status. Load checks template integrity, protocol compatibility, and threshold consistency with calibration evidence.
- Calibrates only on the normal-calibration subset using the fixed 5% rank rule. Insufficient calibration stores JSON null for the infinite threshold and reports predictions as `uncalibrated` / `is_anomaly: null`, avoiding an unsupported normal decision.
- `predict --model ... --image ... --output-name ...` accepts a user image and saves a JSON score/decision, raw model-resolution map, original-resolution map, and labeled orange residual overlay. Output names are validated; existing outputs are preserved. Known official-test paths and reserved image hashes are rejected.
- Explicit input limits: decoded PNG/JPEG/BMP/TIFF, recognized extensions, grayscale/RGB single-frame, 2–8192 pixels per side, at most 32 million pixels. Corrupt inputs, unsupported modes/orientation, and bad dimensions fail clearly. These limits were fixed using software fixtures, not official test results.
- Added `docs/BASELINE.md` and updated README/protocol with working commands, artifact schema, exact scoring/display rules, and alignment/lighting/texture limitations.

### Verification and saved fixture artifacts

- Confirmed no real data at configured `data/mvtec_ad/bottle`; therefore real-data training/calibration could not run. No additional dataset acquisition or official test performance inspection occurred.
- `.venv/bin/python -m pytest -q`: **70 passed**. New checks cover save/load consistency, exact fit-only template statistics, score/map determinism, map geometry, repeated training, calibration changes not altering the template, missing test files not affecting training, known-test prediction guards, corrupt artifacts, threshold tampering, small-calibration behavior, invalid image formats/dimensions/corruption, and end-to-end CLI behavior.
- `.venv/bin/python scripts/verify_fixture_baseline.py --name fixture-baseline-verified`: succeeded. Generates 100 tiny synthetic normal training fixtures (80 fit/20 calibration), synthetic schema-only test records, and a separate synthetic user-image query. This is software verification only, not a defect benchmark. The earlier `fixture-baseline` development run is retained as well.
- Saved fixture model: `artifacts/fixture-baseline-verified/outputs/models/fixture-model/`.
- Saved fixture prediction: `artifacts/fixture-baseline-verified/outputs/predictions/fixture-prediction/`, containing `prediction.json`, both raw maps, and `overlay.png`.
- Visually inspected the generated overlay: readable method/provenance labels, explicit non-probability note, and visible residual highlight. Preview geometry was enlarged for small inputs; raw maps retain documented dimensions.
- `.venv/bin/python -m pip check`: no broken requirements; no new dependencies installed.
- `edge-inspect train --help` / `predict --help`: succeeded. `git check-ignore` confirms model and prediction artifacts are excluded.
- No commits, pushes, or changes to other projects.

### Limitations and next step

This is a functioning classical baseline verified only with synthetic fixtures. Its real defect recall, FPR, localization quality, latency, and memory usage remain unmeasured. Fixed spatial alignment, lighting/reflections, blurred mean appearance, noise-sensitive max scoring, and loss of fine texture at 256×256 limit the method. Overlay intensity is not a defect probability or pixel-level decision. Saved hashes are integrity aids, not proof of dataset authenticity. `evaluate` remains unimplemented; official test performance stays reserved.

Next command for a fresh, clearly labeled software check: `.venv/bin/python scripts/verify_fixture_baseline.py --name my-fixture-check`. Choose an unused name. For real training, first complete official bottle acquisition and run `prepare-data`, then use its manifest with the README's `train` command. Do not run final-test evaluation yet.

## 2026-10-21 — Pretrained PatchCore-style detector

### Implemented and frozen

- Added a CPU **PatchCore-style** detector alongside the unchanged classical scoring method. It uses frozen torchvision ResNet18 `IMAGENET1K_V1`, layer2 features (32×32 spatial positions, 128 components), fixed ImageNet normalization, streamed Algorithm R reservoir sampling, exact chunked Euclidean 1-NN scoring, and bilinear spatial heatmaps.
- `configs/frozen_comparison.json` freezes comparison settings before official evaluation. The bounded 4096-patch reservoir supersedes the original all-fit-patches proposal solely for predictable memory usage; seed 42, backbone/layer, shared data preprocessing, baseline, and normal-calibration rules are retained. No official test performance informed settings.
- Only model-fit normals populate the bank. Retained patch image paths/indices are saved for audit. Calibration occurs after construction, with the same 5% normal-only rank rule. Test images/masks are never opened during training. Default image prediction retains official-test path/hash guards.
- Bank payload is capped at 2 MiB; 64-query × 512-bank chunks bound each returned distance block to 128 KiB. These are exact array-size bounds, not measured total process memory. Framework, backbone, activations, provenance and other allocations add overhead.
- Added `fetch-weights` and `train-features` CLI commands; shared `predict` dispatches from artifact method. Training/load/prediction never implicitly download. Missing weights fail clearly.
- Artifacts contain memory bank, pruned extractor state, source-weight provenance, exact preprocessing/backbone/layers/sampling configuration, fit provenance, calibration scores/evidence, library versions and source/file hashes. NumPy loads disable pickle; torch loads use `weights_only=True`. Saved models support offline CPU inference without the full source-weight cache.
- Extracted the common fit/calibration manifest checks for reuse by both models. Baseline regression tests pass. Feature overlays use a saved calibration-derived display scale, explicitly labeled as distance and not a probability; no query-dependent normalization.
- Documented method simplifications and limits in `docs/PATCHCORE_STYLE.md`. Updated README architecture diagram, optional installation, weight download, training and prediction commands, and the data-protocol amendment.

### Official references and weight status

- Consulted Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022, https://arxiv.org/abs/2106.08265), and the authors' method guide/source repository at https://github.com/amazon-science/patchcore-inspection. Our single-layer compact model and uniform reservoir differ from the original coreset approach; no original benchmark numbers are attributed to this implementation.
- Checked official torchvision feature-extraction, ResNet18 weight definition/source, compatibility table, code license and pretrained-weight notice. Installed optional torch 2.13.0 / torchvision 0.28.0 wheels inside `.venv`; CPU execution verified on this Mac. `requirements-features.lock.txt` records the full resolved feature environment; lightweight `requirements-dev.lock.txt` retained.
- Explicit official download: https://download.pytorch.org/models/resnet18-f37072fd.pth, 46,830,571 bytes. Published SHA-256 prefix verified; full hash `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` recorded in `artifacts/weights/resnet18-source.json`.
- Initial HTTPS fetch failed because this Python installation lacked the needed CA bundle. Added `certifi` to the optional extra and retried with verified TLS; no security bypass or system certificate modification.
- **License limitation:** torchvision code is BSD-3-Clause. No separate explicit license for this weight file was found in the official weight source/definition. Upstream explicitly warns weights can have dataset-derived terms; ImageNet's database access terms are not a blanket weight license. Neither commercial use nor unrestricted redistribution is claimed. Keep weights outside Git and resolve intended publication/deployment rights before redistribution.

### Verification

- `.venv/bin/python -m pip install --no-cache-dir -e '.[dev,features]'`: succeeded, project-local only.
- `.venv/bin/edge-inspect fetch-weights --config configs/default.toml`: succeeded after CA bundle fix. Only pretrained weights downloaded; no MVTec dataset acquired.
- `.venv/bin/python -m pytest -q`: **76 passed**, including actual pretrained-backbone integration, with no skipped tests in this environment. Tests never download; they skip optional feature integration with an explicit reason if the extra/weight cache is absent on a fresh installation.
- New tests: reservoir determinism/cap/provenance; chunked distances against brute force and observed chunk limits; no implicit fetch for missing/bad weights; agreement with the existing pretrained preprocessing route; eval mode/no gradients/CPU; save/load score/map parity; training without any test/mask files; calibration data excluded from bank; changing calibration leaves sampled bank unchanged; user-image prediction and dimensions; changed artifact configuration rejected; frozen comparison matches implementation.
- Ran CLI `train-features` against the existing synthetic fixture manifest under `artifacts/fixture-baseline-verified`, with run name `patchcore-style-v1`, `--data-kind synthetic_fixture`, and the official weight file. Then ran shared `predict` on `synthetic-user-input.png`. Both succeeded. 80 fit fixtures streamed 81,920 patches into 4,096 bank rows; 20 separate normal fixtures provided a finite calibration threshold. These counts describe software fixtures, not real dataset verification.
- Saved model: `artifacts/fixture-baseline-verified/outputs/models/patchcore-style-v1/`.
- Saved prediction/maps/overlay: `artifacts/fixture-baseline-verified/outputs/predictions/patchcore-style-v1/`. Visually inspected the labeled overlay; score/map outputs are fixture-only checks, not benchmark results.
- `pip check`: no broken requirements. `train-features --help` succeeds. `git check-ignore` confirms official weights and saved extractor weights are excluded.
- No commits, pushes, or modifications to other projects.

### Remaining work / next step

Real MVTec bottle data remains unavailable locally. Real fit/calibration runs, model-artifact digest freezes, final image/pixel evaluation, hardware speed/RSS benchmarks, and a full demo remain pending. Both configurations are frozen, but no real model quality has been established. Weight-specific licensing remains unresolved for publication/commercial use; source and limitations are documented without assuming permission.

Next verification command: `.venv/bin/python -m pytest -q`. Once official bottle data is acquired and manifested, use the README's `train` and `train-features` commands on the identical frozen split. Freeze their resulting artifacts/thresholds before enabling any final evaluation. `evaluate` remains explicitly unimplemented.

## 2026-10-22 — Frozen evaluation implementation; official run pending

### Implemented

- Replaced the evaluation placeholder with a frozen two-model comparison. Preflight validates provenance, common manifest, exact fit/calibration membership, frozen settings, model/runtime versions and test inventory. Both model artifacts and calibration thresholds are recorded, with manifest/configuration/code hashes, hardware and protocol, before test inference.
- Added a persistent local held-out-attempt ledger. Official reruns must be labeled exploratory; failures retain partial logs and an attempt record. This is an audit aid, not a tamper-proof provenance system.
- Exact tie-aware image AUROC; strict calibrated-threshold confusion counts, recall and FPR; class/sample counts; per-type descriptive results; exact original-resolution per-image pixel AUROC with equal-image macro aggregation. This is explicitly not pooled pixel AUROC. Empty/one-class inputs and unavailable finite thresholds produce null with reasons.
- CPU timing captures each image's model-only and validation/decode/preprocess/model durations after five fit-normal warmups, batch size one, 256×256, one torch thread. Snapshot and report state all measurement boundaries, ordering and cache limitations. No total-memory benchmark implemented.
- JSON/JSONL results, raw maps, comparison and score plots, deterministic median-score success/failure overlays, and a report with error/localization observations and clearly labeled hypotheses. Official feature-vs-baseline comparison reports whether image AUROC improved, without changing configurations.
- Added the optional Matplotlib evaluation extra and `requirements-evaluation.lock.txt`. Kept existing lightweight/feature locks. Updated README and `docs/EVALUATION.md`; `docs/final_evaluation_status.json` records the absent official results explicitly.

### Verification and evidence

- Installed `.[dev,features,evaluation]` in the project venv; Matplotlib 3.11.2. `pip check` reports no broken requirements. No system-wide packages, dataset downloads, commits or pushes.
- Ran `.venv/bin/python scripts/verify_fixture_evaluation.py --name fixture-evaluation-verified`: completed. Generated 100 synthetic training fixtures, split 80 fit / 20 calibration, trained both unchanged frozen methods, evaluated six synthetic test fixtures. Real pretrained weights were already cached; no download occurred. All outputs are labeled synthetic software verification, not MVTec results.
- Saved fixture snapshot, results, plots and report under `artifacts/fixture-evaluation-verified/outputs/evaluations/comparison/`. Inspected the comparison plot and PatchCore-style FN overlay visually. The fixture run exercises false positives, a missed synthetic defect, and successful predictions; it supplies no evidence of real model quality or superiority.
- Known-answer tests cover perfect/reversed/tied/0.75 AUROC, ROC integration, invalid/absent classes, strict threshold ties, confusion counts, undefined calibration, equal-image pixel weighting and latency summaries. Evaluation tests cover missing data, provenance refusal, repeat-attempt ledger preservation, deterministic representative selection, and end-to-end pre-inference snapshot creation with uncalibrated outputs.

### Limitations and next step

The user confirmed official MVTec bottle data is not yet acquired. **Final real held-out evaluation could not be run.** No real model comparison, official metrics, real-data latency or real missed-defect/false-positive analysis is claimed. Dataset acquisition/terms, real fit/calibration, and subsequent one-time frozen evaluation remain required. Weight-license limitations from the previous milestone remain applicable. Memory benchmarking and the full local demo remain future work.

Next available command: `.venv/bin/python -m pytest -q`. To generate a fresh software-only report, run `.venv/bin/python scripts/verify_fixture_evaluation.py --name my-evaluation-check` with an unused name. For the actual benchmark, first acquire official bottle data through the documented official flow; then follow README prepare/train/train-features/evaluate commands without changing frozen settings based on test results.

Final checks: `.venv/bin/python -m pytest -q` → **94 passed in 36.58s**, no skips in this environment. `compileall` and `evaluate --help` succeeded. `git check-ignore` confirms generated evaluation results and pretrained weights remain excluded. Only Project 2 files were edited.

## 2026-10-23 — ONNX feature extractor and measured CPU comparison

### Implemented

- Identified the reliable export boundary: saved frozen ResNet18 through layer2 only. Added `export-onnx` using the PyTorch dynamo exporter, fixed FP32 normalized `[1,3,256,256]` input, `[1,128,32,32]` output, opset 18, embedded tensor data, ONNX full checker and seeded generated-input parity check. No weights/data download or retraining.
- Export metadata binds the graph hash to the exact source model metadata and extractor hash, feature configuration, versions, shape/opset and source license limitation. Exported files remain ignored. Re-export overwrites are rejected.
- Added optional `predict --onnx` and a shared external `score_patches` method. Pillow preprocessing, NumPy normalization/layout, exact torch nearest-neighbor scoring, map resize and threshold remain outside ONNX. The bank, score definition, frozen model settings and original calibration threshold are unchanged; this is not a complete ONNX detector. The default runtime remains original PyTorch.
- Added `benchmark-onnx`: default inputs are normal-calibration images only. A matching completed frozen evaluation is required to opt into already-exposed test examples, explicitly labeled exploratory. Model/manifest/split binding and export integrity are checked. No threshold calibration or model fitting occurs.
- Before inference, records source/model/export/input hashes, calibration evidence, hardware/runtime versions, thread settings, provider, tolerances, warmup/repeats and code hashes. Separate sequential worker processes measure component and end-to-end p50/p95 latencies with raw repetitions retained. Five warmups use a model-fit normal; input batch one, 256×256 FP32; torch and ORT intra/inter-op threads one; CPUExecutionProvider, sequential ORT, full graph optimization, spinning disabled.
- Fixed embedding/score/map tolerances atol=rtol=1e-4 plus exact decision comparisons. Tests explicitly detect a decision flip near the threshold even if scores pass numerical tolerance. No retuning hides changed decisions.
- Process memory includes psutil stage RSS and resource process-lifetime peak RSS, with platform unit conversion. Methodology explicitly includes process startup/imports/load/cache/parity-array saving/pixel metrics; it is not incremental model allocation or GPU/system RAM. Timing excludes those non-inference operations.
- Added optional ONNX/export/runtime/psutil dependencies and `requirements-onnx.lock.txt`; README, `docs/ONNX.md`, `docs/ONNX_RESULTS.md`, and compact `docs/onnx_results.json` document setup, measurements and limitations. Original frozen evaluation outputs remain unchanged.

### Actual measurements (software fixtures, not MVTec benchmarks)

- Hardware: actual Apple M2 Mac, 8 CPU cores, 16 GB unified RAM, arm64 macOS 27.0.1, Python 3.14.0. No Raspberry Pi, other edge device, MPS or GPU benchmark.
- Installed project-locally: ONNX 1.23.2, onnxscript 0.7.2, ONNX Runtime 1.31.0, psutil 7.2.2; retained torch 2.13.0/torchvision 0.28.0. Official exporter/runtime/threading/psutil documentation consulted; references in docs/ONNX.md. Network-restricted initial package-index query failed; authorized project-local package installation succeeded. No system-wide install.
- Export command: `.venv/bin/edge-inspect export-onnx --config artifacts/fixture-evaluation-verified/config.toml --model artifacts/fixture-evaluation-verified/outputs/models/features --run-name resnet18-layer2-fp32`. Succeeded; generated-input maximum embedding error about 3.19e-6.
- Both benchmark commands used the fixture config above, its manifest `outputs/manifests/bottle-5125e2af4ff2e40213050d8d12677d318c35406be9991e0531e45f06fb9f0f01.json`, model `outputs/models/features`, export `outputs/exports/resnet18-layer2-fp32`, `--warmup 5`. Paths are relative to `artifacts/fixture-evaluation-verified/`.
- Calibration run: `--run-name onnx-normal-calibration --repeats 5`; 20 fixture normals × 5 passes. Original end-to-end p50/p95 **270.128/275.477 ms**, ONNX path **287.216/289.549 ms**. Process-lifetime peak RSS approximately **377.5 vs 349.4 MiB**. Embedding p50 13.804 vs 30.629 ms; external scoring about 256 ms dominated both paths. Parity passed; **0/20 decision changes**. No defect AUROC/recall is defined for normal-only inputs.
- Exposed-fixture run: `--run-name onnx-exposed-fixtures --repeats 10 --evaluation artifacts/fixture-evaluation-verified/outputs/evaluations/comparison`; 6 fixtures × 10 passes. Original end-to-end p50/p95 **270.163/272.511 ms**, ONNX path **287.397/293.782 ms**. Process peaks **378.8 vs 349.7 MiB**. **0/6 decision changes**, all numerical checks passed. The synthetic quality metrics agree across runtimes; they establish no real defect-detection quality.
- ONNX was about 6.3–6.4% slower at median end-to-end latency in these workloads, despite lower observed whole-worker peak RSS. Do not assume ONNX accelerates this pipeline. Fixed ordering, correlated samples, cache/thermal effects and system load limit generalization. Tests did not run concurrently with the two recorded benchmarks.
- Full JSON/raw repeated samples/embeddings/maps/snapshots/logs/reports are under `artifacts/fixture-evaluation-verified/outputs/benchmarks/{onnx-normal-calibration,onnx-exposed-fixtures}`. Export under `outputs/exports/resnet18-layer2-fp32`. Files were not published.

### Verification / next step

- `.venv/bin/python -m pytest -q`: **98 passed**, one upstream exporter FutureWarning, 46.12 seconds. New tests cover normalization/input validation, threshold-flip reporting independent of allclose, actual export/backbone/score/map parity, ONNX prediction, benchmark subprocess execution and memory/latency records, uncalibrated/undefined quality behavior, model binding and corrupt export rejection. Tests use synthetic fixtures and explicitly cached weights; never download.
- `pip check`: no broken requirements. Source/CLI compilation and help checked. No new quantization: any future attempt must use normal-only calibration and report quality tradeoffs separately.
- **Real-data parity, defect quality and runtime measurement remain pending official bottle acquisition and real training.** Pretrained-weight licensing remains as previously documented. A full demo remains pending. No commits or pushes; changes confined to Project 2.
- Next command: `.venv/bin/python -m pytest -q`. After acquiring official data and training the frozen feature model, use docs/ONNX.md to export it and benchmark calibration inputs before any test quality comparison. Keep original inference as the default based on current measured results.

## 2026-10-23 — Local Streamlit demo

Implemented app.py and edge_inspect.demo using actual saved baseline/feature artifacts, discovered inside project artifacts. Includes upload/type/size/decode/dimension validation, held-out hash guard, temporary-file cleanup, model selector, loading/errors/missing-model guidance, score/threshold/decision, original/heatmap/overlay and explicit single-call timing boundaries. Loopback default, telemetry off; no camera or deployment. Original geometric sample has no third-party content and is dedicated to CC0; no dataset photos redistributed. Fixture provenance is prominent; no predictions fabricated. Model load occurs outside timing; reruns clear stale results.

Installed Streamlit 1.66.1 project-locally; official file-uploader documentation consulted. Smoke tests cover corrupt/type/oversized uploads, saved-artifact scoring and Streamlit result rendering. Browser verified baseline sample inference, model selection and feature inference; actual screenshots retained under docs/screenshots and linked in README. Initial sandbox refused loopback binding; approved loopback-only server startup succeeded. README includes exact install/start commands and sample rights. No commits/pushes. Real-data validation remains pending; arbitrary uploaded images are not validated inspection use cases.

Final verification: full suite **100 passed**, one upstream ONNX exporter FutureWarning, 47.07 seconds. Browser observed actual feature score 6.13346 against saved threshold 3.45517 on the synthetic geometry sample (not benchmark quality). Local server remains available at http://127.0.0.1:8501. Resolved environment recorded in requirements-demo.lock.txt.

## 2026-10-23 — Portfolio release preparation

Added standalone-root lightweight CI (Python 3.11/3.14; baseline/data/metrics/demo tests, no dataset/weights), CPU Dockerfile and Compose setup with allowlisted build context, read-only trusted artifact mount, non-root execution and host-loopback port. Docker CLI exists but daemon socket is absent: Compose config validates, image build and runtime remain unverified. CI is inactive in the current multi-project parent Git layout until this folder becomes the repo root or publication explicitly arranges a parent workflow. No parent files changed.

Added model card, dataset/method/weight attribution, deployment guide, concise case study, two-minute recording script and docs/RELEASE.md checklist. README now leads with evidence boundaries, measured optimization table, portable fixture quickstart and release links; stale “demo pending” claims and machine-specific startup paths corrected. Retained original evaluation/runtime reports because scoring, preprocessing and thresholds did not change. No new quality or performance claims. Camera experiments remain absent.

Added scripts/audit_release.py for Git-visible file audit (credential patterns, machine paths, large files, model/archive extensions and unreviewed images). Reviewed actual geometry-only screenshots; no dataset photos. Ignored artifact trees deliberately excluded; never release a whole workspace archive. Source license remains an owner decision. Pretrained weight redistribution/commercial rights remain unresolved. Dataset acquisition and real frozen evaluation remain blocked by absent official data.

Verification: full suite 100 passed in 47.23 s, one upstream ONNX exporter FutureWarning. Added and ran explicit missing-model UI smoke test afterward; demo suite 3 passed. Lightweight command, config/pip checks and Compose validation recorded in release checklist. No commits/pushes or other project modifications. Next step: resolve source-code licensing and acquire real data; Docker/remote CI verification remain unchecked. See docs/RELEASE.md for concrete gates before publication.

Release check detail: lightweight CI selection **85 passed in 2.53 s** before the additional missing-model test; `check-config`, `pip check`, `compileall` and `docker compose config --quiet` succeeded. Final Git-visible audit: **64 paths, zero findings**. Remote CI and Docker image/runtime verification remain pending.

## Publication preflight — awaiting source-license decision

User authorized a separate public `edge-inspect` repository. Inspected Git layout: Project 2 has no nested repository and currently resides in the multi-project parent repo; no parent Git files or other projects modified. Network-enabled `gh auth status` confirms authenticated account `thilankathushan` with repo/workflow scopes. The requested repository lookup returned 404, so no existing repository will be overwritten. Initial sandbox-only authentication probe was misleading; network-enabled verification succeeded. Git-visible source audit: 64 files, zero flagged findings. Dataset/model artifacts remain excluded. Asked owner for required source-code license choice before creating/publishing the repository; no license assumed. Publication remains pending that decision.

Publication preflight full suite: **101 passed**, one upstream exporter FutureWarning, 44.04 seconds. Authentication requires no user repair. No commit, repository creation or push performed while license choice is pending.

## Authorized standalone publication

Owner delegated the code-license decision; selected MIT for original source, preserving separate dataset/weight terms. Created a nested standalone Git repository only inside Project 2 on main, with author Thilanka Thushan <158076487+thilankathushan@users.noreply.github.com>. Parent Git metadata, other projects, ignored datasets/models and local artifacts remain untouched. Rechecked GitHub authentication and confirmed requested repository absent before creation. Staged 65 source/test/documentation/config/license/permitted screenshot files; release audit found zero issues. Last full suite: 101 passed. No scoring changes; existing measurements reused. Docker context includes LICENSE for packaging. Public demo hosting is not configured; the demo remains local. Publication authorized with absent real-data evaluation and unverified Docker runtime clearly disclosed.

## Publication completed

Public repository: https://github.com/thilankathushan/edge-inspect . Initial source release commit d7e7c32 pushed to main, using the requested author identity and no co-author trailers or AI credits. Network verification confirms PUBLIC visibility. Remote lightweight CI passed on Linux Python 3.11 and 3.14 in run 39992327495, including configuration and release audit. No force-push, overwritten repository, other-project edits, dataset/model upload, or publicly hosted demo. Local working tree was clean following the initial push. This documentation follow-up records completion; real-data evaluation, Docker runtime verification and weight redistribution rights remain as disclosed. Initial publication attempt was not executed because approval review hit a usage limit; the user requested continuation and the normal reviewed retry succeeded.

## 2026-10-25 — Bottle mirror acquired and validated

User authorized bottle-only download after the publisher's MyDrive link failed. Downloaded 357 files (156,570,806 bytes) from foersben/mvtec-ad pinned at c75b39616f84db43677bcc8228caaafaf5096d7f. All transfers matched mirror LFS SHA-256 or regular Git-blob digests; full SHA-256 receipt retained. Included CC BY-NC-SA 4.0 license and MVTec attribution preserved. This is third-party provenance, not independently authenticated publisher data.

Ran `.venv/bin/python artifacts/acquisition/download_bottle.py` and `.venv/bin/python artifacts/acquisition/validate_bottle.py`. Structural/decode/dimension/binary-mask/pairing/duplicate checks passed. 209 normal training images split deterministically into 167 fit and 42 calibration; all 83 test images and 63 masks reserved. No inference, test metrics, visual galleries or tuning performed. Manifest includes actual mirror/revision, acquisition time and receipt hash; exact digest/path recorded in docs/DATA_ACQUISITION.md. Updated local README notice, protocol acquisition amendment and machine-readable evaluation status. Existing fixture models and measurements unchanged.

Data lives under ignored data/mvtec_ad/bottle; receipt, inventory and acquisition scripts under ignored artifacts/acquisition. No commit/push or dataset redistribution in this task. Next: train/calibrate both frozen methods on the new manifest, then snapshot and run final evaluation with mirror provenance explicitly disclosed.

## 2026-10-25 — Real-data evaluation and release completion

Completed the frozen bottle comparison with the settings recorded in docs/REAL_EVALUATION_PROTOCOL.md before first test inference. Source: 357 files / 156,570,806 bytes from foersben/mvtec-ad at pinned revision c75b39616f84db43677bcc8228caaafaf5096d7f. Mirror digest and structural checks passed, but source authenticity is not established against the MVTec publisher. The dataset and trained artifacts remain ignored/outside Git. Split: 167 fit normals, 42 normal calibration images, 83 held-out test images (20 normal, 63 defects). Thresholds were frozen at baseline 0.3748110830783844 and PatchCore-style 3.768542766571045.

Frozen results: baseline image AUROC 0.8595, recall 46/63 (73.02%), FPR 1/20 (5%), confusion TP/FN/FP/TN=46/17/1/19, macro per-image original-resolution pixel AUROC 0.8375. PatchCore-style image AUROC 0.9460, recall 58/63 (92.06%), FPR 2/20 (10%), counts 58/5/2/18, pixel macro AUROC 0.9679 over the 63 anomalous images; the 20 normal images are excluded because their per-image pixel AUROC is undefined. No tuning occurred after viewing results. Per-type outcomes, representative errors and their local overlays are documented in artifacts/evaluations/frozen-bottle-v1/REPORT.md. Public aggregate report and portable hash summary are docs/REAL_RESULTS.md and docs/final_evaluation_status.json; restricted images remain local. Test split is exposed, so all subsequent comparisons are labeled exploratory.

Real-data ONNX benchmark completed separately: M2/macOS 27.0.1/Python 3.14.0 CPU, FP32 256×256 batch one, one thread per runtime, five fit-normal warmups. On 42 normal calibration images × 5 repeats, original vs ONNX end-to-end p50/p95 was 291.272/295.594 vs 311.132/320.937 ms; decision flips 0/42. Process-lifetime peak RSS 392.2 vs 375.0 MiB (whole worker, not model-only). On the already exposed 83-image test split × 3 repeats, p50/p95 was 295.904/329.006 vs 315.163/343.648 ms; decision flips 0/83; AUROC/recall/FPR/pixel macro are unchanged within reported precision. This is an exploratory runtime parity check, not an untouched quality evaluation. The exported graph contains only ResNet18 through layer2. Original PyTorch remains the faster default. Full raw records and reports are under ignored artifacts/benchmarks.

Demo model discovery now prioritizes real-data models and visibly explains their third-party provenance. Browser inspection ran actual saved baseline and feature inference on the generated CC0 geometry sample; displayed results were computed live. Captured final screenshot set and a short silent screenshot-sequence GIF use only the synthetic sample; they are not MVTec imagery or a continuous video. Demo remains local on loopback; this workspace serves it on port 8503 to avoid the other project at 8501.

Verification: `.venv/bin/python -m pytest -q` **103 passed**, one upstream torch ONNX exporter FutureWarning (52.37 s). `pip check` passed. Release audit found zero issues across 75 Git-visible paths at the last scan; reviewed screenshots/assets contain generated shapes/UI/plots only. Both arm64 Linux containers built: baseline image sha256:c72959e6252a9c779d917f7f2e7459d4037c942151d3a578c43b87e56114963d; feature image sha256:9a00218e12bba7ee762ae0e92b02303c5615bbc5dcf3631065a98b112b26f052. Their direct smoke runs passed actual saved-model scoring, invalid/corrupt/oversized upload rejection, uid 10001 and read-only artifact checks. Image inspection confirmed no dataset/model paths baked into the image. A later health-endpoint check could not complete because Docker Desktop became manually paused; report it as unverified until resumed. Native browser app and AppTest are verified.

Release checklist: [x] both detectors trained on fit-only real normals; [x] normal-only threshold freeze; [x] first real held-out comparison and error analysis saved; [x] real ONNX calibration and exposed-test parity/speed measurements; [x] local UI exercised with both real artifact choices and generated sample; [x] source/data/weight license distinctions and mirror source limitation documented; [x] offline CI test selection strengthened with fixture-trained demo inference; [x] arm64 baseline/feature image builds and direct model smoke tests; [x] screenshot/GIF and portfolio docs; [x] no secret/dataset/model files in Git-visible release audit. Remaining gates/limits: third-party mirror authenticity is not independently confirmed; pretrained weight redistribution/commercial rights are unresolved and no weights are published; container HTTP health check needs Docker Desktop resumed; no continuous narrated recording, camera experiments, quantization, Raspberry Pi, other hardware, public hosting, production/security/load validation. The frozen test set is spent; any later quality improvement is exploratory and requires a new confirmatory test set.

Next: `docker compose up -d` after resuming Docker Desktop, then `curl -fsS http://127.0.0.1:8501/_stcore/health`; the already verified native demo command is `.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8503`.
