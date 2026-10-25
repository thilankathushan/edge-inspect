"""Classical spatial RGB mean-template detector; no deep learning."""
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np

from .config import Config, ValidationError
from .data import sha256_file
from .manifest import select_records
from .preprocessing import iter_samples
from .protocol import PROTOCOL, normal_threshold

METHOD = 'rgb-mean-template-v1'
SHAPE = (3, 256, 256)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except (OSError, ValueError) as exc:
        raise ValidationError(f'Cannot read JSON {path}: {exc}') from exc


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def output_directory(root: Path, name: str) -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', name):
        raise ValidationError('Output name must be 1–80 letters, digits, underscores or hyphens')
    destination = root / name
    if destination.exists():
        raise ValidationError(f'Output already exists; choose another name: {destination}')
    return destination


def validate_tensor(image: np.ndarray):
    if not isinstance(image, np.ndarray) or image.shape != SHAPE or image.dtype != np.float32 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValidationError('Expected finite RGB float32 tensor (3,256,256) in [0,1]')


@dataclass
class TemplateModel:
    template: np.ndarray
    threshold: float
    metadata: dict

    def score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        validate_tensor(image)
        residual = image - self.template
        anomaly_map = np.mean(residual * residual, axis=0, dtype=np.float32)
        return float(anomaly_map.max()), anomaly_map

    def save(self, destination: Path):
        validate_tensor(self.template)
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination / 'template.npy', self.template, allow_pickle=False)
        metadata = {**self.metadata, 'template_sha256': sha256_file(destination / 'template.npy')}
        write_json(destination / 'metadata.json', metadata)

    @classmethod
    def load(cls, destination: Path):
        metadata = read_json(destination / 'metadata.json')
        try:
            if metadata['schema_version'] != 1 or metadata['method'] != METHOD or metadata['protocol'] != PROTOCOL:
                raise ValidationError('Unsupported model method or preprocessing protocol')
            if metadata['template_sha256'] != sha256_file(destination / 'template.npy'):
                raise ValidationError('Model template hash mismatch')
            template = np.load(destination / 'template.npy', allow_pickle=False)
            validate_tensor(template)
            scores = [r['score'] for r in metadata['calibration']['scores']]
            if any(not isinstance(s, (int, float)) or not 0 <= s <= 1 for s in scores):
                raise ValidationError('Invalid calibration scores in model artifact')
            if metadata['calibration']['count'] != len(scores) or metadata['calibration']['target_fpr'] != PROTOCOL['target_fpr']:
                raise ValidationError('Inconsistent calibration metadata')
            threshold = normal_threshold(scores, PROTOCOL['target_fpr'])
            expected_status = 'calibrated' if math.isfinite(threshold) else 'insufficient_calibration'
            if metadata['calibration']['status'] != expected_status:
                raise ValidationError('Inconsistent calibration status')
            stored = metadata['calibration']['threshold']
            if stored != (threshold if math.isfinite(threshold) else None):
                raise ValidationError('Stored threshold does not match calibration evidence')
            if metadata['data_kind'] not in {'synthetic_fixture', 'official_mvtec_ad'}:
                raise ValidationError('Invalid data provenance label')
            return cls(template, threshold, metadata)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f'Invalid model artifact {destination}: {exc}') from exc


def train(config: Config, manifest_path: Path, run_name: str, data_kind: str) -> dict:
    if data_kind not in {'synthetic_fixture', 'official_mvtec_ad'}:
        raise ValidationError('Declare data_kind as synthetic_fixture or official_mvtec_ad')
    destination = output_directory(config.artifact_root / 'models', run_name)
    manifest = read_json(manifest_path)
    try:
        fit, calibration, test_hashes = training_records(config, manifest)
        total = np.zeros(SHAPE, dtype=np.float64)
        for _, image, _ in iter_samples(manifest, config.data_root, 'model_fit'):
            total += image
        model = TemplateModel((total / len(fit)).astype(np.float32), math.inf, {})
        scores = []
        for row, image, _ in iter_samples(manifest, config.data_root, 'normal_calibration'):
            score, _ = model.score(image)
            scores.append({'path': row['path'], 'sha256': row['sha256'], 'score': score})
        model.threshold = normal_threshold([r['score'] for r in scores])
        finite = math.isfinite(model.threshold)
        model.metadata = {
            'schema_version': 1, 'method': METHOD, 'category': config.category,
            'protocol': PROTOCOL, 'seed': config.seed,
            'data_kind': data_kind, 'provenance_note': 'User-declared input origin, not independently authenticated',
            'manifest_sha256': sha256_file(manifest_path),
            'dataset_acquisition': manifest.get('acquisition'),
            'versions': {'python': platform.python_version(), 'numpy': version('numpy'), 'pillow': version('Pillow')},
            'implementation_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'fit_count': len(fit), 'fit_records': [{'path': r['path'], 'sha256': r['sha256']} for r in fit],
            'reserved_test_sha256': test_hashes,
            'calibration': {'count': len(scores), 'target_fpr': PROTOCOL['target_fpr'],
                            'rank': math.ceil((len(scores) + 1) * (1 - PROTOCOL['target_fpr'])),
                            'threshold': model.threshold if finite else None,
                            'status': 'calibrated' if finite else 'insufficient_calibration',
                            'comparison': 'score > threshold', 'scores': scores,
                            'ties_at_threshold': sum(r['score'] == model.threshold for r in scores)},
            'evaluation': 'No official test performance evaluated',
        }
        model.save(destination)
        return {'model': str(destination), 'data_kind': data_kind, 'fit_count': len(fit),
                'calibration_count': len(scores), 'threshold': model.metadata['calibration']['threshold'],
                'status': model.metadata['calibration']['status']}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f'Invalid training manifest: {exc}') from exc


def training_records(config: Config, manifest: dict):
    fit = select_records(manifest, 'model_fit')
    calibration = select_records(manifest, 'normal_calibration')
    if manifest['category'] != config.category or manifest['seed'] != config.seed:
        raise ValidationError('Configuration category/seed differs from the frozen manifest')
    if not fit or not calibration:
        raise ValidationError('Both model_fit and normal_calibration must be nonempty')
    # Validate the frozen role allocation against the prespecified hash split.
    normal = fit + calibration
    ordered = sorted(normal, key=lambda r: (hashlib.sha256(f"{config.seed}\n{Path(r['path']).relative_to(config.category).as_posix()}".encode()).hexdigest(), r['path']))
    n_cal = math.ceil(len(normal) * PROTOCOL['calibration_fraction'])
    if {r['path'] for r in ordered[:n_cal]} != {r['path'] for r in calibration}:
        raise ValidationError('Manifest roles differ from the prespecified split')
    if len({r['sha256'] for r in normal}) != len(normal):
        raise ValidationError('Duplicate content across fit/calibration records')
    test_hashes = [r['sha256'] for r in manifest['records'] if r['role'] == 'final_test']
    if set(test_hashes) & {r['sha256'] for r in normal}:
        raise ValidationError('Fit/calibration content overlaps final test')
    return fit, calibration, test_hashes
