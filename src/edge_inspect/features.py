"""Explicit torchvision weights and frozen CPU feature extraction."""
from pathlib import Path
import os
from urllib.request import urlopen
import tempfile
import ssl
import numpy as np
from .baseline import validate_tensor, write_json
from .config import ValidationError
from .data import sha256_file
from .protocol import PROTOCOL

WEIGHT_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
WEIGHT_NAME = 'resnet18-f37072fd.pth'
WEIGHT_LICENSE = 'No separate weight-specific license verified; torchvision code BSD-3-Clause; upstream warns of dataset-derived terms'
FEATURE_CONFIG = {
    'version': 'patchcore-style-resnet18-v1', 'backbone': 'resnet18',
    'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'feature_layers': ['layer2'],
    'embedding_dim': 128, 'patch_grid': [32, 32], 'patch_aggregation': 'none; layer2 receptive fields',
    'memory_bank_capacity': 4096, 'sampling': 'Algorithm R reservoir; NumPy PCG64; row-major patches',
    'sampling_seed': 42, 'query_chunk': 64, 'bank_chunk': 512,
    'distance': 'Euclidean 1-NN, torch.cdist direct', 'image_score': 'maximum raw patch distance',
    'map_resize': 'bilinear align_corners=False to 256x256; no smoothing',
    'device': 'cpu', 'batch_size': 1, 'torch_threads': 1,
    'preprocessing': {'size': [256, 256], 'resize': 'Pillow bilinear direct no crop',
                      'input': 'RGB float32 CHW /255',
                      'mean': PROTOCOL['feature_mean'], 'std': PROTOCOL['feature_std']},
}


def dependencies():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except (ImportError, RuntimeError) as exc:
        raise ValidationError("Feature detector requires compatible optional dependencies: .venv/bin/python -m pip install -e '.[dev,features]'") from exc


def fetch_weights(directory: Path) -> dict:
    """The sole network operation in the detector; invoked explicitly by CLI."""
    try:
        import certifi
    except ImportError as exc:
        raise ValidationError('Install the features extra to obtain the CA bundle') from exc
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHT_NAME
    if not target.exists():
        fd, temp = tempfile.mkstemp(prefix='.weights-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as output, urlopen(WEIGHT_URL, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 64 * 1024 * 1024:
                        raise ValidationError('Unexpected weight download size exceeds 64 MiB')
                    output.write(block)
            if not sha256_file(Path(temp)).startswith('f37072fd'):
                raise ValidationError('Official weight SHA-256 prefix mismatch')
            os.replace(temp, target)
        finally:
            Path(temp).unlink(missing_ok=True)
    digest = check_official_weights(target)
    receipt = {'source': WEIGHT_URL, 'sha256': digest, 'bytes': target.stat().st_size,
               'weights': FEATURE_CONFIG['weights'], 'license_status': WEIGHT_LICENSE}
    write_json(directory / 'resnet18-source.json', receipt)
    return {**receipt, 'path': str(target)}


def check_official_weights(path: Path) -> str:
    if not path.is_file():
        raise ValidationError(f'Missing pretrained weights: {path}. Run fetch-weights explicitly; training never downloads.')
    digest = sha256_file(path)
    if not digest.startswith('f37072fd'):
        raise ValidationError('Expected official ResNet18 IMAGENET1K_V1 weights (SHA-256 prefix f37072fd)')
    return digest


class Extractor:
    def __init__(self, state_path: Path, *, official: bool):
        torch, torchvision = dependencies()
        from torchvision.models.feature_extraction import create_feature_extractor
        torch.set_num_threads(FEATURE_CONFIG['torch_threads'])
        if official:
            check_official_weights(state_path)
        # No model builder receives a weights enum: implicit downloads are impossible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            backbone = torchvision.models.resnet18(weights=None)
        try:
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            if official:
                backbone.load_state_dict(state, strict=True)
            self.network = create_feature_extractor(backbone, return_nodes={'layer2': 'patches'}).cpu().eval()
            if not official:
                self.network.load_state_dict(state, strict=True)
            self.network.requires_grad_(False)
        except (RuntimeError, ValueError, OSError) as exc:
            raise ValidationError(f'Invalid backbone state: {exc}') from exc

    def embeddings(self, image: np.ndarray) -> np.ndarray:
        validate_tensor(image)  # Public input is identical raw preprocessing for both models.
        torch, _ = dependencies()
        mean = np.asarray(FEATURE_CONFIG['preprocessing']['mean'], dtype=np.float32)[:, None, None]
        std = np.asarray(FEATURE_CONFIG['preprocessing']['std'], dtype=np.float32)[:, None, None]
        normalized = np.ascontiguousarray((image - mean) / std)
        with torch.inference_mode():
            features = self.network(torch.from_numpy(normalized).unsqueeze(0))['patches']
        if tuple(features.shape) != (1, 128, 32, 32):
            raise ValidationError('Unexpected backbone feature geometry')
        patches = features[0].permute(1, 2, 0).reshape(-1, 128).contiguous().numpy().copy()
        if not np.isfinite(patches).all():
            raise ValidationError('Nonfinite patch embeddings')
        return patches

    def save(self, path: Path):
        torch, _ = dependencies()
        torch.save(self.network.state_dict(), path)
