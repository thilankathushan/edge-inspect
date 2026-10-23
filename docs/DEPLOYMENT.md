# Local CPU deployment

## Verified native path

From the project root, create a Python 3.14 environment and install `.[dev,demo]` for the baseline or `.[dev,demo,features]` for the learned detector. Run `streamlit run app.py --server.address 127.0.0.1`. The README quickstart gives full commands. Model artifacts remain separate from source and are discovered under artifacts/**/models/*. Model loads happen on each explicit inspection. No internet access or automatic weights download is required for inference with existing artifacts.

## Docker candidate (not runtime-verified)

The Docker CLI is available, but its daemon was unavailable during preparation. Compose configuration validates; image build, Linux wheel availability, health check and inference are **unverified**. Do not substitute the native Mac measurements for container measurements.

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

Before claiming container support: build both desired variants; check healthy status; run baseline/feature inference with the generated sample; check corrupt and oversized uploads; verify artifacts are not writable and no dataset/weights were baked into layers. No public-hosting, TLS/authentication, load handling or production security claims are made.
