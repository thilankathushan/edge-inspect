# Two-minute demo walkthrough

The repository includes actual UI screenshots and an animated screenshot walkthrough, using only the original CC0 geometry sample. It is a silent sequence of captured UI states, not a continuous screen recording or real bottle quality demonstration. A narrated recording is optional.

1. Start `.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8503`. Open http://127.0.0.1:8503. This alternate port preserves other local portfolio apps. Record only this app, with unrelated windows/notifications hidden.
2. Say: “Edge Inspect compares a classical baseline with a CPU PatchCore-style anomaly detector. These models were trained on real bottle images from a documented third-party MVTec mirror.” Show the provenance note and category limitation.
3. Select `models/bottle-baseline-v1`, check **Use a synthetic geometry sample**, and run inspection. Explain: “This generated shape demonstrates the interface; it is not a bottle or evidence of defect quality.” Show the actual score, threshold and decision without substituting scripted values.
4. Show original, heatmap and overlay. Say: “Orange means distance from the normal reference, not defect probability. Separate normal calibration images set the threshold. This single-call timer excludes model loading and visualization.”
5. Select `models/bottle-patchcore-v1`, run the same sample, and show the new output. A different-looking map does not establish better detection.
6. Show `docs/REAL_RESULTS.md`: “On 83 frozen test images, image AUROC improved from 0.8595 to 0.9460 and detected defects from 46 to 58 of 63. False alarms increased from one to two of 20 normal images.” Explain that the small sample limits generalization.
7. Show `docs/REAL_ONNX_RESULTS.md`: “Only the feature extractor is ONNX. On this M2 calibration workload, end-to-end median was about 311 ms versus 291 ms for the original. Search dominates; ONNX was not faster.” Show decision parity and the measurement boundaries.
8. End with the model card: “Local portfolio prototype, not production validation. No camera or edge-board results; data authenticity and redistribution limitations are explicit.”

The first frozen evaluation must not be rerun as untouched for recording. Use the preserved report. Do not publish MVTec photos or trained weights with the walkthrough. On a new checkout without real models, use the README offline fixture setup and clearly describe that different provenance.
