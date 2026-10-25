# Bottle mirror acquisition — verified 2026-10-25

The user reported that the original MVTec bottle link redirects to a MyDrive “Page not found” page, then explicitly authorized downloading the bottle-only Hugging Face mirror.

- Canonical dataset: original MVTec AD, bottle; MVTec Software GmbH.
- Actual acquisition source: https://huggingface.co/datasets/foersben/mvtec-ad
- Pinned repository revision: `c75b39616f84db43677bcc8228caaafaf5096d7f`.
- Selected directory: `bottle/` only; no other categories or complete-dataset archive.
- Mirror inventory: 357 files, 156,570,806 bytes, including `license.txt` and `readme.txt`.
- Supplied notices identify Copyright 2019 MVTec Software GmbH and CC BY-NC-SA 4.0; they retain the original paper attribution. Original bytes and notices are preserved. MIT licensing of project code does not apply to these data.
- This is a **third-party mirror**, not a publisher-authenticated download. Matching mirror SHA-256/LFS or Git blob digests verifies transfer integrity relative to the pinned mirror, not agreement with an authoritative MVTec archive. Do not call this an independently authenticated official acquisition.

Local ignored evidence is in `artifacts/acquisition/`: pinned tree metadata, acquisition script, copied notices and per-file receipt. Dataset bytes are under `data/mvtec_ad/bottle`. At acquisition time only integrity and structural validation was performed. Subsequent authorized training and frozen evaluation are recorded separately in REAL_RESULTS.md.

The original publisher reference in manifest `source` identifies the dataset, not this download host. Use the provenance-enriched manifest identified below for future real-data training. Its `acquisition` field records the actual mirror/revision and receipt hash. Existing fixture artifacts and published fixture measurements are unchanged. Any later real-data report must disclose mirror provenance and its authentication limit.

## Completed validation

All 357 file transfers matched pinned mirror digests (SHA-256 for LFS objects; Git blob SHA-1 for regular repository files). Full local SHA-256 hashes are retained in the receipt. Image decoding, dimensions, binary/nonempty masks, pairings, duplicate detection and layout checks passed. The two non-image files are preserved license/attribution notices.

- 209 normal training images: **167 model-fit / 42 normal-calibration** with unchanged seed 42.
- 83 held-out test images: 20 normal, 63 anomalous, with 63 masks.
- Test defect counts: broken_large 20; broken_small 22; contamination 21.
- Manifest: `artifacts/manifests/bottle-e1bbf78c4b1ecd2dbc5f0394877dfc02c753efb31ea5fcdd8b358154e60ff546.json`.
- Manifest SHA-256: `e1bbf78c4b1ecd2dbc5f0394877dfc02c753efb31ea5fcdd8b358154e60ff546`.
- Acquisition receipt SHA-256: `2fca23bd066aeaadc37a35eb79e2ffae18c0ac1b7630070ccdc3b590c02e48ed`.

No test inference or visual model selection occurred during acquisition. The later frozen real-data evaluation is complete; see REAL_RESULTS.md. Dataset files remain outside Git.

## Acquiring the same bottle directory elsewhere

The author already has these files locally; no additional download is needed there. For a new checkout, prefer the publisher when its download works. If using the same third-party mirror under the dataset terms, the official [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) supports revision-pinned selective downloads:

```sh
.venv/bin/python -m pip install huggingface_hub
.venv/bin/hf download foersben/mvtec-ad --repo-type dataset \
  --revision c75b39616f84db43677bcc8228caaafaf5096d7f \
  --include 'bottle/**' --local-dir data/mvtec_ad
.venv/bin/edge-inspect inspect-data --config configs/default.toml
.venv/bin/edge-inspect prepare-data --config configs/default.toml
```

This command is documented from the official CLI, not the transfer method used for the recorded acquisition (which checked each pinned mirror digest with a local downloader). Preserve license.txt/readme.txt and record the mirror/revision with your experiment. The author’s enriched manifest also contains its acquisition receipt hash; your manifest can therefore have a different digest even with identical image bytes. Do not describe mirror transfer verification as publisher authentication.
