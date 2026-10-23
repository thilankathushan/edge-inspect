# Edge Inspect: making anomaly detection inspectable

Visual inspection often has many normal examples and few representative defects. Edge Inspect explores a normal-only detector that explains a high image score spatially, while making calibration and runtime costs visible. The project targets one bottle category rather than claiming broad visual quality control.

I built a reproducible path from validated image/mask manifests to a classical RGB-template baseline, then a compact PatchCore-style detector. Normal training data is split before fitting; calibration sets thresholds but never populates the feature bank. Frozen configurations and pre-run artifact hashes preserve the distinction between a held-out evaluation and later exploration. Undefined metrics are reported explicitly.

The learned branch trades original PatchCore coreset complexity for a bounded, seeded 4096-patch reservoir and chunked exact nearest-neighbor search. It runs on CPU, saves provenance, and produces a coarse spatial map. The baseline remains useful as a transparent reference that exposes sensitivity to alignment and lighting.

Optimization produced an instructive negative result: on my Apple M2, the FP32 ONNX backbone path was slower on synthetic calibration inputs—287.2 ms versus 270.1 ms median end-to-end—although whole-worker peak RSS was lower. Search dominated both pipelines at roughly 256 ms. Exporting one component did not remove the main runtime cost. Embedding/score/map parity passed, with no threshold decision changes across 26 fixtures; the original runtime therefore remains the default.

These are measured optimization observations, not MVTec accuracy results. Official data is not yet acquired, so real defect recall, false alarms and localization quality remain unknown. The local Streamlit demo uses actual saved fixture-trained models and visibly states that limitation. No camera experiment or edge-board benchmark has been performed.

The deliverable is an inspectable engineering prototype: input validation, deterministic splits, explicit weight downloads, reproducible artifacts, transparent metric definitions, a working demo, and offline test coverage. Remaining release gates are real data and frozen evaluation, license decisions, and container verification—not unsupported production-readiness claims. See ONNX_RESULTS.md for measurement boundaries and MODEL_CARD.md for intended use.
