# Local CPU deployment

## Verified native path

From the project root, create a Python 3.14 environment and install `.[dev,demo]` for the baseline or `.[dev,demo,features]` for the learned detector. Run `streamlit run app.py --server.address 127.0.0.1`. The README quickstart gives full commands. Model artifacts remain separate from source and are discovered under artifacts/**/models/*. Model loads happen on each explicit inspection. No internet access or automatic weights download is required for inference with existing artifacts.

## Verified Docker CPU path

On 2026-10-25 both baseline and feature images built for arm64 Linux and passed direct inference, invalid-upload, non-root and read-only artifact smoke checks. On 2026-10-26, after Docker Desktop resumed, `edge-inspect-release-check` reported `running healthy` and `GET http://127.0.0.1:8502/_stcore/health` returned `ok`. This container publishes host loopback port 8502 to container port 8501. The Compose default below uses host port 8501. Native Mac latency is not container latency.

From this project directory:

```sh
mkdir -p artifacts
docker compose build
docker compose up -d
# http://127.0.0.1:8501
docker compose ps
docker compose logs demo
docker compose down
```

Default image supports baseline inference. For learned artifacts: `FEATURES=true docker compose build` then `docker compose up -d`. This installs CPU PyTorch wheels from the official CPU index, but never model weights. This optional image is larger and requires verification on the target architecture. The mutable base-image tag and dependency ranges are not a bit-reproducible supply-chain lock; record image digest/package versions after a successful build.

The allowlisted Docker context includes only source, application configuration and packaging files. The host artifacts directory is mounted read-only, with no data directory mounted. Container runs as non-root with dropped capabilities, read-only root, and a writable temporary filesystem; port 8501 is published on host loopback only. Internal binding to 0.0.0.0 is for container networking, not authorization to expose the host publicly. Stop the native demo first if it already occupies 8501. Mount only trusted, rights-cleared model artifacts readable by UID 10001. No artifacts means the UI shows setup guidance.

Those checks passed for the recorded arm64 images. Validate other deployment architectures separately. These Docker checks do not establish load handling or production security. The separate Render smoke check is recorded below.

## Free Render public demo

The owner authorized a labeled generated-fixture baseline with the separately recorded real bottle results. `render.yaml` selects a **Free** Python web service, installs `.[demo]` without PyTorch, and generates the synthetic baseline during build. `EDGE_INSPECT_PUBLIC_DEMO=1` displays the recorded evaluation table and limits model selection to synthetic RGB baselines. No real training data, model weights or private artifacts are uploaded or fetched. Uploaded images are processed on Render, which the UI explains; use non-sensitive samples.

From Render, create a Blueprint using this public repository and `render.yaml`, or create a Python Web Service with:

- Repository: https://github.com/thilankathushan/edge-inspect ; branch `main`.
- Plan: **Free** (do not choose a paid plan).
- Python version: `3.14.0`.
- Build: `pip install '.[demo]' && python scripts/prepare_public_demo.py`.
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.maxUploadSize 10`.
- Environment: `EDGE_INSPECT_PUBLIC_DEMO=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Health endpoint: `/_stcore/health`.

After deployment verify the health endpoint, public provenance warning, recorded result table and a synthetic-sample inspection. The host assigns the public URL; do not assume a service-name URL is available before deployment. No disks, database, secret or payment credentials are needed by this configuration. A provider may impose account verification independently.

As checked 2026-10-26, [Render Free](https://render.com/docs/free) sleeps after 15 minutes without traffic. The [Free compute plan](https://render.com/docs/compute-plans) provides 0.1 CPU and 512 MB RAM. First visits after sleep can be slow; concurrent uploads may hit resource limits. This host is not benchmarked by the local M2 results. Configuration uses the [official Blueprint fields](https://render.com/docs/blueprint-spec). The owner deployed [the public demo](https://edge-inspect-demo.onrender.com). On 2026-10-26, HTTPS health returned `ok`; browser checks confirmed the fixture warning, real-results table, and a successful generated-sample inspection with original, heatmap and overlay. Sample score 0.108578 exceeded threshold 0.00942338 (Flagged). This is a functional smoke check, not a quality or latency benchmark. No Render load, security or uptime testing is claimed.
