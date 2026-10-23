"""Reproducible manifest snapshots and leakage-safe role selection."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from importlib.metadata import version
from .config import Config, ValidationError
from .data import scan_data, sha256_file
from .protocol import PROTOCOL


def build_manifest(config: Config) -> dict:
    if config.category != 'bottle':
        raise ValidationError('Protocol bottle-v1 supports only the original MVTec AD bottle category')
    rows, summary = scan_data(config)
    test = [row for row in rows if row['official_split'] == 'test']
    if not any(r['is_anomaly'] for r in test) or not any(not r['is_anomaly'] for r in test):
        raise ValidationError('Complete category required: test must contain normals and masked anomalies')
    train = [row for row in rows if row['official_split'] == 'train']
    if len(train) < 2:
        raise ValidationError('At least two normal training images are needed for fit/calibration')
    def key(row):
        relative = Path(row['path']).relative_to(config.category).as_posix()
        return hashlib.sha256(f'{config.seed}\n{relative}'.encode()).hexdigest(), relative
    ordered = sorted(train, key=key)
    n_calibration = math.ceil(len(train) * PROTOCOL['calibration_fraction'])
    calibration = {row['path'] for row in ordered[:n_calibration]}
    for row in rows:
        row['role'] = ('final_test' if row['official_split'] == 'test' else
                       'normal_calibration' if row['path'] in calibration else 'model_fit')
    # Absolute machine paths and timestamps are deliberately excluded from the snapshot.
    summary.pop('root')
    return {'schema_version': 1, 'dataset': 'MVTec AD (original)',
            'source': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
            'license': 'CC-BY-NC-SA-4.0', 'category': config.category, 'seed': config.seed,
            'protocol': PROTOCOL.copy(), 'pillow_version': version('Pillow'),
            'numpy_version': version('numpy'), 'summary': summary, 'records': rows}


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_manifest(config: Config) -> dict:
    manifest = build_manifest(config)  # Validate everything before creating output.
    payload = manifest_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    directory = config.artifact_root / 'manifests'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'{config.category}-{digest}.json'
    fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'manifest': str(destination), 'sha256': digest,
            'counts': {role: sum(r['role'] == role for r in manifest['records'])
                       for role in ('model_fit', 'normal_calibration', 'final_test')}}


def select_records(manifest: dict, role: str, *, allow_final_test=False) -> list[dict]:
    if role not in {'model_fit', 'normal_calibration', 'final_test'}:
        raise ValidationError(f'Unknown dataset role: {role}')
    if role == 'final_test' and not allow_final_test:
        raise ValidationError('Final test loading requires explicit allow_final_test=True after freezing models and thresholds')
    if manifest.get('schema_version') != 1 or manifest.get('protocol') != PROTOCOL:
        raise ValidationError('Manifest schema or preprocessing protocol differs from this implementation')
    rows = manifest['records']
    seen = set()
    for row in rows:
        if row['path'] in seen:
            raise ValidationError('Duplicate manifest path')
        seen.add(row['path'])
        if row['role'] in {'model_fit', 'normal_calibration'}:
            valid = row['official_split'] == 'train' and row['label'] == 'good' and not row['is_anomaly'] and row['mask_path'] is None
        elif row['role'] == 'final_test':
            valid = (row['official_split'] == 'test'
                     and row['is_anomaly'] == (row['label'] != 'good')
                     and (row['mask_path'] is not None) == row['is_anomaly'])
        else:
            valid = False
        if not valid:
            raise ValidationError(f'Invalid split/role assignment: {row["path"]}')
        parts = Path(row['path']).parts
        if len(parts) != 4 or parts[:3] != (row['category'], row['official_split'], row['label']):
            raise ValidationError(f'Path disagrees with manifest split: {row["path"]}')
    return [row.copy() for row in rows if row['role'] == role]


def verified_path(root: Path, relative: str, digest: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValidationError(f'Unsafe manifest path: {relative}')
    root = root.resolve()
    candidate = root / path
    if root not in candidate.resolve().parents or any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root and root in p.parents):
        raise ValidationError(f'Unsafe manifest path: {relative}')
    if sha256_file(candidate) != digest:
        raise ValidationError(f'File changed since manifest creation: {relative}')
    return candidate
