# Portfolio release checklist

Prepared 2026-10-23. **Publication authorized; production readiness is not claimed.**

## Complete locally

- [x] Existing foundation, model, evaluation, ONNX and demo code inspected; full suite passed (100 tests before adding the no-model smoke test; demo suite then passed 3 tests).
- [x] Lightweight CI workflow prepared for Python 3.11/3.14 with no dataset/weight downloads. Selected test command verified locally; remote runners not executed.
- [x] Native CPU demo and configuration smoke checks; missing-model state tested without fabricated predictions.
- [x] Dockerfile, allowlisted context and Compose configuration added; configuration validates. Container artifacts are mounted read-only, not baked into the image.
- [x] Model card, attribution, case study, demo script, exact commands and actual screenshots included.
- [x] Existing measured optimization and fixture evaluation reports retained. No detector/preprocessing/threshold changes made for this release; rerunning quality or speed measurements is unnecessary.
- [x] Git-visible release audit: no detected credentials, home-directory paths, archives, weights or unreviewed images. Screenshots contain only generated geometry, local UI and actual fixture output. No restricted dataset photos included. Heuristic scan cannot prove secret absence.
- [x] Git layout checked: project is inside a multi-project parent repo; no Git mutations or other project edits made.

## Blocked or unverified

- [x] Owner delegated license selection; MIT applied to original source. Third-party terms remain separate.
- [ ] Resolve pretrained-weight and derived model redistribution rights before sharing artifacts. Code BSD licensing does not establish weight rights. Dataset CC BY-NC-SA terms are separate.
- [ ] Acquire official bottle data under its terms; record acquisition details, train/calibrate both frozen models and perform real held-out evaluation. Real accuracy and localization remain unknown.
- [ ] Docker daemon available; build and run desired variants, verify CPU wheels and health/inference on target Linux architecture. Daemon socket was absent, so no successful image-build/runtime claim is made.
- [x] Standalone repository initialized inside Project 2; workflow is at its repository root. Parent repository and other projects preserved.
- [ ] Run CI remotely after authorized publication. Python 3.11/Linux matrix and clean dependency resolution are not locally verified by the Mac's Python 3.14 run.
- [ ] Record demo video using DEMO_SCRIPT.md; only screenshots currently exist.
- [x] Public publication explicitly authorized by owner; source-only audit required before commit. No force-push or existing repository overwrite permitted.

## Exclusions and scope

Ignored local artifacts include real pretrained weights, fixture model states, runtime exports and diagnostic arrays. They are not release source files. Never distribute the entire workspace directory as a zip. Keep source packaging separate from ignored artifact trees. Data/model licensing and host-specific absolute paths may remain inside private ignored provenance snapshots; public summaries use portable paths.

Benchmark findings: no real MVTec benchmark exists. Optimization findings: measured M2 CPU runtimes on synthetic fixtures, ONNX slower with lower whole-worker RSS. Camera experiments: none. Production reliability, public-service security, container performance and other-device claims: unverified.
