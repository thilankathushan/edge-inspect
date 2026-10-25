"""Bounded, CPU-only PatchCore-style detector (not the original algorithm)."""
from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path
import platform
import numpy as np

from .baseline import read_json, write_json, output_directory, training_records
from .config import Config, ValidationError
from .data import sha256_file
from .features import Extractor, FEATURE_CONFIG, WEIGHT_URL, WEIGHT_LICENSE, check_official_weights, dependencies
from .preprocessing import iter_samples
from .protocol import PROTOCOL, normal_threshold

METHOD = 'patchcore-style-resnet18-v1'


class Reservoir:
    """Algorithm R: uniform bounded sample over the ordered stream of fit patches."""
    def __init__(self, capacity: int, dimension: int, seed: int):
        if not 1 <= capacity <= FEATURE_CONFIG['memory_bank_capacity'] or dimension != 128:
            raise ValidationError('Reservoir must have 1..4096 rows and 128 columns')
        self.bank = np.empty((capacity, dimension), dtype=np.float32)
        self.origins = [None] * capacity
        self.seen = 0
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def add(self, patches: np.ndarray, path: str):
        if patches.ndim != 2 or patches.shape[1] != 128 or patches.dtype != np.float32 or not np.isfinite(patches).all():
            raise ValidationError('Invalid patch embeddings')
        for patch_index, vector in enumerate(patches):
            self.seen += 1
            slot = self.seen - 1 if self.seen <= len(self.bank) else int(self.rng.integers(self.seen))
            if slot < len(self.bank):
                self.bank[slot] = vector
                self.origins[slot] = {'path': path, 'patch_index': patch_index}

    def finish(self):
        count = min(self.seen, len(self.bank))
        if count == 0:
            raise ValidationError('Cannot build an empty memory bank')
        return self.bank[:count].copy(), self.origins[:count]


def nearest_distances(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Never allocate an all-query/all-bank matrix or a Q×B×D difference tensor."""
    torch, _ = dependencies()
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != 128 or bank.shape[1] != 128 or not len(bank):
        raise ValidationError('Nearest-neighbor inputs must be nonempty 128-dimensional matrices')
    if len(bank) > FEATURE_CONFIG['memory_bank_capacity'] or not np.isfinite(query).all() or not np.isfinite(bank).all():
        raise ValidationError('Invalid or oversized nearest-neighbor inputs')
    results = np.empty(len(query), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(query), FEATURE_CONFIG['query_chunk']):
            q = torch.from_numpy(np.ascontiguousarray(query[start:start + FEATURE_CONFIG['query_chunk']], dtype=np.float32))
            best = torch.full((len(q),), float('inf'))
            for offset in range(0, len(bank), FEATURE_CONFIG['bank_chunk']):
                b = torch.from_numpy(np.ascontiguousarray(bank[offset:offset + FEATURE_CONFIG['bank_chunk']], dtype=np.float32))
                distance = torch.cdist(q[None], b[None], p=2, compute_mode='donot_use_mm_for_euclid_dist')[0]
                best = torch.minimum(best, distance.min(dim=1).values)
            results[start:start + len(q)] = best.numpy()
    return results


@dataclass
class PatchModel:
    extractor: Extractor
    bank: np.ndarray
    threshold: float
    metadata: dict

    def score(self, image: np.ndarray):
        return self.score_patches(self.extractor.embeddings(image))

    def score_patches(self, patches):
        """Unchanged external nearest-neighbor scoring and map interpolation."""
        torch, _ = dependencies()
        distances = nearest_distances(patches, self.bank)
        patch_map = distances.reshape(32, 32)
        with torch.inference_mode():
            anomaly_map = torch.nn.functional.interpolate(torch.from_numpy(patch_map)[None, None], size=(256, 256), mode='bilinear', align_corners=False)[0, 0].numpy().copy()
        return float(distances.max()), anomaly_map

    def save(self, destination: Path):
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination / 'memory_bank.npy', self.bank, allow_pickle=False)
        self.extractor.save(destination / 'extractor.pth')
        write_json(destination / 'metadata.json', {**self.metadata,
            'bank_sha256': sha256_file(destination / 'memory_bank.npy'),
            'extractor_sha256': sha256_file(destination / 'extractor.pth')})

    @classmethod
    def load(cls, destination: Path, *, onnx_path: Path | None = None):
        metadata = read_json(destination / 'metadata.json')
        try:
            if metadata['schema_version'] != 1 or metadata['method'] != METHOD or metadata['feature_config'] != FEATURE_CONFIG or metadata['protocol'] != PROTOCOL:
                raise ValidationError('Unsupported PatchCore-style artifact configuration')
            for name, key in [('memory_bank.npy', 'bank_sha256'), ('extractor.pth', 'extractor_sha256')]:
                if sha256_file(destination / name) != metadata[key]:
                    raise ValidationError(f'Artifact hash mismatch: {name}')
            bank = np.load(destination / 'memory_bank.npy', allow_pickle=False)
            if not isinstance(bank, np.ndarray) or bank.ndim != 2 or bank.shape[1] != 128 or not 1 <= len(bank) <= FEATURE_CONFIG['memory_bank_capacity'] or bank.dtype != np.float32 or not np.isfinite(bank).all():
                raise ValidationError('Invalid memory bank')
            fit_paths = {r['path'] for r in metadata['fit_records']}
            calibration = metadata['calibration']
            if fit_paths & {r['path'] for r in calibration['scores']}:
                raise ValidationError('Calibration overlaps model-fit provenance')
            if len(metadata['bank_origins']) != len(bank) or any(r['path'] not in fit_paths or not 0 <= r['patch_index'] < 1024 for r in metadata['bank_origins']):
                raise ValidationError('Invalid memory-bank provenance')
            scores = [r['score'] for r in calibration['scores']]
            threshold = normal_threshold(scores)
            if any(s < 0 for s in scores) or calibration['count'] != len(scores) or calibration['target_fpr'] != PROTOCOL['target_fpr'] or calibration['threshold'] != (threshold if math.isfinite(threshold) else None):
                raise ValidationError('Invalid calibration evidence')
            expected_status = 'calibrated' if math.isfinite(threshold) else 'insufficient_calibration'
            if calibration['status'] != expected_status:
                raise ValidationError('Invalid calibration status')
            if onnx_path is None:
                extractor = Extractor(destination / 'extractor.pth', official=False)
            else:
                from .onnx_backend import OnnxExtractor
                dependencies()[0].set_num_threads(FEATURE_CONFIG['torch_threads'])
                extractor = OnnxExtractor(onnx_path, destination)
            return cls(extractor, bank, threshold, metadata)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f'Invalid PatchCore-style artifact: {exc}') from exc


def train_features(config: Config, manifest_path: Path, run_name: str, data_kind: str, weights: Path) -> dict:
    if data_kind not in {'synthetic_fixture', 'official_mvtec_ad'}:
        raise ValidationError('Declare valid data provenance')
    if config.seed != FEATURE_CONFIG['sampling_seed']:
        raise ValidationError('Frozen comparison requires seed 42; changes require a new protocol')
    destination = output_directory(config.artifact_root / 'models', run_name)
    manifest = read_json(manifest_path)
    try:
        fit, calibration, test_hashes = training_records(config, manifest)
        source_hash = check_official_weights(weights)
        extractor = Extractor(weights, official=True)
        reservoir = Reservoir(FEATURE_CONFIG['memory_bank_capacity'], 128, config.seed)
        for row, image, _ in iter_samples(manifest, config.data_root, 'model_fit'):
            reservoir.add(extractor.embeddings(image), row['path'])
        bank, origins = reservoir.finish()
        model = PatchModel(extractor, bank, math.inf, {})
        scores = []
        for row, image, _ in iter_samples(manifest, config.data_root, 'normal_calibration'):
            score, _ = model.score(image)
            scores.append({'path': row['path'], 'sha256': row['sha256'], 'score': score})
        model.threshold = normal_threshold([r['score'] for r in scores])
        finite = math.isfinite(model.threshold)
        model.metadata = {
            'schema_version': 1, 'method': METHOD, 'protocol': PROTOCOL, 'feature_config': FEATURE_CONFIG,
            'category': config.category, 'seed': config.seed, 'data_kind': data_kind,
            'provenance_note': 'User-declared dataset origin; official torchvision pretrained weights verified by hash prefix',
            'weights': {'source': WEIGHT_URL, 'sha256': source_hash, 'license_status': WEIGHT_LICENSE},
            'manifest_sha256': sha256_file(manifest_path),
            'dataset_acquisition': manifest.get('acquisition'),
            'versions': {name: version(name) for name in ['numpy', 'Pillow', 'torch', 'torchvision']},
            'python_version': platform.python_version(),
            'implementation_sha256': {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'fit_count': len(fit), 'fit_records': [{'path': r['path'], 'sha256': r['sha256']} for r in fit],
            'bank_origins': origins, 'patches_seen': reservoir.seen, 'bank_rows': len(bank), 'bank_bytes': bank.nbytes,
            'reserved_test_sha256': test_hashes,
            'calibration': {'scores': scores, 'count': len(scores), 'target_fpr': PROTOCOL['target_fpr'],
                            'rank': math.ceil((len(scores) + 1) * (1 - PROTOCOL['target_fpr'])),
                            'threshold': model.threshold if finite else None,
                            'status': 'calibrated' if finite else 'insufficient_calibration',
                            'comparison': 'score > threshold',
                            'ties_at_threshold': sum(r['score'] == model.threshold for r in scores)},
            'evaluation': 'No official test performance evaluated',
        }
        model.save(destination)
        return {'model': str(destination), 'method': METHOD, 'data_kind': data_kind,
                'bank_rows': len(bank), 'bank_bytes': bank.nbytes, 'patches_seen': reservoir.seen,
                'fit_count': len(fit), 'calibration_count': len(scores),
                'threshold': model.metadata['calibration']['threshold'], 'status': model.metadata['calibration']['status']}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f'Invalid feature training manifest: {exc}') from exc
