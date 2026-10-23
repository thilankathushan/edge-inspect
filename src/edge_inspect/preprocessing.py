"""Deterministic transformations shared by future training and inference."""
from pathlib import Path
import numpy as np
from PIL import Image
from .data import validate_image
from .manifest import select_records, verified_path
from .protocol import PROTOCOL


def preprocess_image(path: Path, *, pretrained=False) -> np.ndarray:
    validate_image(path)
    with Image.open(path) as image:
        image = image.convert('RGB').resize(tuple(PROTOCOL['image_size']), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
    if pretrained:
        pixels = (pixels - np.asarray(PROTOCOL['feature_mean'], dtype=np.float32)) / np.asarray(PROTOCOL['feature_std'], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def preprocess_mask(path: Path) -> np.ndarray:
    validate_image(path, mask=True)
    with Image.open(path) as mask:
        mask = mask.convert('L').resize(tuple(PROTOCOL['image_size']), Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def iter_samples(manifest: dict, data_root: Path, role='model_fit', *, pretrained=False, allow_final_test=False):
    """Stream one sample at a time; verify frozen hashes before each decode."""
    for row in select_records(manifest, role, allow_final_test=allow_final_test):
        path = verified_path(data_root, row['path'], row['sha256'])
        mask = None
        if row['mask_path'] is not None:
            mask_path = verified_path(data_root, row['mask_path'], row['mask_sha256'])
            mask = preprocess_mask(mask_path)
        yield row, preprocess_image(path, pretrained=pretrained), mask
