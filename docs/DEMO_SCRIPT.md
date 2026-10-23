# Two-minute recording script

1. Start the local app using the README. Close unrelated windows and notifications. Record only the app; do not show credentials, private file paths or unrelated uploads.
2. Say: “Edge Inspect is a local visual anomaly-detection prototype. These models are fixture-trained; real MVTec evaluation is pending.” Show the provenance warning.
3. Select a baseline artifact and the synthetic geometry sample. Explain that the sample is generated, not a bottle photograph or benchmark. Click Run inspection; show the actual score, calibration threshold and decision.
4. Show original, heatmap and overlay. Say: “Orange indicates anomaly strength, not defect probability. The threshold is selected only from separate normal calibration images.” Explain single-call timing excludes model loading and is not a benchmark.
5. Select the learned artifact, then run the same sample. Show the newly computed output. Do not imply that a more attractive map proves better defect accuracy. If no learned artifact exists, omit this segment and state that prerequisite.
6. Open docs/ONNX_RESULTS.md. Say: “On my M2, ONNX was slower end-to-end on fixtures despite lower process RSS. The feature extractor alone is exported; nearest-neighbor scoring remains external.” Show 270.1 vs 287.2 ms and the measurement boundaries.
7. End with the model card/release checklist. Say: “Real held-out quality, container validation and licensing decisions remain explicit next steps.” Do not demonstrate a webcam or present arbitrary images as validated inspection.

Use actual live outputs; do not replace a score with a scripted value. Rehearse before recording. The recording itself has not been produced by this task.
