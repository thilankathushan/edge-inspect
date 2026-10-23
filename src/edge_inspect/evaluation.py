"""Frozen held-out comparison. Snapshot first; never train or recalibrate here."""
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter_ns

import numpy as np
from PIL import Image

from .baseline import TemplateModel, output_directory, read_json, training_records, write_json
from .config import Config, ValidationError
from .data import EXTENSIONS, sha256_file, validate_image
from .features import FEATURE_CONFIG
from .manifest import select_records, verified_path
from .metrics import auroc, summarize
from .preprocessing import preprocess_image
from .protocol import PROTOCOL

EVALUATION_PROTOCOL = {
    'version': 'heldout-comparison-v1',
    'image_auroc': 'Exact pairwise ranking with half credit for ties; positive=defect',
    'decision': 'score > saved normal-calibration threshold; no recalibration',
    'pixel_auroc': 'Exact within-image at original mask resolution; unweighted mean over images with both pixel classes; not pooled',
    'map_alignment': 'Pillow float32 bilinear 256x256 anomaly map to original dimensions; no normalization',
    'undefined': 'JSON null plus reason; no finite threshold means no operating decisions/counts',
    'per_type': 'Each defect type versus all shared test normals; AUROC requires >=2 of each; counts/recall whenever defined',
    'latency': {'device': 'cpu', 'image_size': [256, 256], 'batch_size': 1, 'torch_threads': 1,
                'warmup': 5, 'warmup_input': 'first model_fit normal; excluded from timing',
                'repetitions_per_image': 1, 'timer': 'perf_counter_ns',
                'model_only': 'model.score on prepared tensor, including feature normalization/search/map interpolation',
                'pipeline': 'validation/decode/resize/normalization plus model.score; includes repeated decoder validation',
                'excluded': 'model load, file hashing, masks, original-size map resize, metrics, output writes and plots',
                'order': 'baseline then patchcore, each in manifest path order; OS caches uncontrolled; no cold-start claim'},
    'representatives': 'one median-score example per TP/FN/FP/TN/uncalibrated bucket, path tie-break; absent buckets reported',
    'selection': 'No tuning or retraining after evaluation; later reruns explicitly exploratory',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hardware_info():
    info = {'os': platform.platform(), 'architecture': platform.machine(),
            'python': platform.python_version(), 'logical_cpu_count': os.cpu_count(),
            'cpu': platform.processor() or None, 'memory': None, 'device': 'cpu'}
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(['system_profiler', 'SPHardwareDataType', '-json'], capture_output=True, text=True, timeout=15, check=True)
            hardware = json.loads(result.stdout)['SPHardwareDataType'][0]
            # Deliberately exclude serial numbers and hardware identifiers.
            info['cpu'] = hardware.get('chip_type', hardware.get('cpu_type'))
            info['memory'] = hardware.get('physical_memory')
            info['cpu_description'] = hardware.get('number_processors')
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            info['hardware_note'] = 'Detailed hardware query unavailable'
    return info


def artifact_hashes(directory):
    names = ('metadata.json', 'template.npy') if (directory / 'template.npy').exists() else ('metadata.json', 'extractor.pth', 'memory_bank.npy')
    return {name: sha256_file(directory / name) for name in names}


def prepare_evaluation(config, manifest_path, baseline_path, feature_path, comparison_path, fixture):
    if not (config.data_root / config.category).is_dir():
        raise ValidationError('Official dataset is unavailable at the configured path. Acquire bottle data and train real fit/calibration models before final evaluation; fixtures are not real results.')
    manifest = read_json(manifest_path)
    comparison = read_json(comparison_path)
    if comparison.get('data_protocol') != PROTOCOL or comparison.get('feature_model') != FEATURE_CONFIG or comparison.get('baseline') != {'method': 'rgb-mean-template-v1', 'score': 'max mean-channel squared residual', 'template': 'model_fit mean RGB'} or config.seed != comparison.get('seed'):
        raise ValidationError('Comparison differs from frozen configurations')
    fit, calibration, test_hashes = training_records(config, manifest)
    test = sorted(select_records(manifest, 'final_test', allow_final_test=True), key=lambda r: r['path'])
    if not test:
        raise ValidationError('No held-out test records in manifest')
    provenance = 'synthetic_fixture' if fixture else 'official_mvtec_ad'
    manifest_hash = sha256_file(manifest_path)
    frozen = {}
    for label, path, method in [('baseline', baseline_path, 'rgb-mean-template-v1'), ('patchcore_style', feature_path, 'patchcore-style-resnet18-v1')]:
        metadata = read_json(path / 'metadata.json')
        if metadata.get('data_kind') != provenance:
            raise ValidationError('Model provenance does not match evaluation mode; synthetic models require --fixture and cannot produce official results')
        if metadata.get('method') != method or metadata.get('protocol') != PROTOCOL or metadata.get('category') != config.category or metadata.get('seed') != config.seed:
            raise ValidationError(f'{label} differs from the frozen comparison')
        if metadata.get('manifest_sha256') != manifest_hash:
            raise ValidationError(f'{label} was fitted/calibrated using a different manifest')
        fit_expected = sorted((r['path'], r['sha256']) for r in fit)
        cal_expected = sorted((r['path'], r['sha256']) for r in calibration)
        if sorted((r['path'], r['sha256']) for r in metadata['fit_records']) != fit_expected or sorted((r['path'], r['sha256']) for r in metadata['calibration']['scores']) != cal_expected:
            raise ValidationError('Saved fit/calibration provenance disagrees with frozen manifest')
        if sorted(metadata.get('reserved_test_sha256', [])) != sorted(test_hashes):
            raise ValidationError('Saved test reservation differs from manifest')
        if label == 'patchcore_style' and metadata.get('feature_config') != FEATURE_CONFIG:
            raise ValidationError('Feature configuration is not frozen')
        for name, expected in metadata['versions'].items():
            if name != 'python' and version(name) != expected:
                raise ValidationError(f'Runtime {name} differs from trained artifact; restore the recorded environment before evaluation')
        frozen[label] = {'path': str(path.resolve()), 'files': artifact_hashes(path), 'metadata': metadata}
    # Check integrity of fit/calibration sources, without opening any test bytes.
    for row in fit + calibration:
        verified_path(config.data_root, row['path'], row['sha256'])
    expected_paths = {r['path'] for r in test} | {r['mask_path'] for r in test if r['mask_path']}
    found_paths = {p.relative_to(config.data_root).as_posix() for part in ['test', 'ground_truth']
                   for p in (config.data_root / config.category / part).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if found_paths != expected_paths:
        raise ValidationError('Test/mask inventory differs from frozen manifest; do not silently omit samples')
    # Bind the held-out ledger to test bytes independently of training/config changes.
    test_identity = sorted((r['path'], r['sha256'], r['mask_sha256']) for r in test)
    test_digest = hashlib.sha256(json.dumps(test_identity).encode()).hexdigest()
    return manifest, comparison, frozen, test, test_digest


def reserve_holdout(root, digest, record, *, exploratory=False):
    ledger = root / 'heldout-ledger'
    ledger.mkdir(parents=True, exist_ok=True)
    target = ledger / f'{digest}.json'
    if exploratory:
        return
    try:
        with target.open('x') as stream:
            json.dump(record, stream, indent=2)
    except FileExistsError as exc:
        raise ValidationError('This held-out test set already has an evaluation attempt. Preserve the record; later runs require --exploratory and cannot be called untouched.') from exc


def evaluate(config: Config, manifest_path: Path, baseline_path: Path, feature_path: Path,
             comparison_path: Path, run_name: str, *, fixture=False, exploratory=False):
    from .reporting import reporting_ready, create_report
    reporting_ready(config.artifact_root)
    try:
        manifest, comparison, frozen, test, digest = prepare_evaluation(config, manifest_path, baseline_path, feature_path, comparison_path, fixture)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f'Invalid evaluation inputs: {exc}') from exc
    mode = 'synthetic_fixture_verification' if fixture else ('exploratory' if exploratory else 'official_heldout')
    destination = output_directory(config.artifact_root / 'evaluations', run_name)
    from .patchcore import PatchModel
    models = {'baseline': TemplateModel.load(baseline_path), 'patchcore_style': PatchModel.load(feature_path)}
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = {'recorded_before_test_inference_utc': utc_now(), 'mode': mode,
                'dataset_origin': 'User-declared provenance; not independent publisher authentication',
                'config': config.as_dict(), 'manifest_sha256': sha256_file(manifest_path),
                'test_identity_sha256': digest, 'comparison': comparison,
                'comparison_sha256': sha256_file(comparison_path), 'evaluation_protocol': EVALUATION_PROTOCOL,
                'models': frozen, 'hardware': hardware_info(),
                'runtime_versions': {name: version(name) for name in ['numpy', 'Pillow', 'torch', 'torchvision', 'matplotlib']},
                'evaluation_code_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(destination / 'prerun_snapshot.json', snapshot)
    shutil.copyfile(manifest_path, destination / 'manifest.json')
    if not fixture:
        reserve_holdout(config.artifact_root / 'evaluations', digest, {'mode': mode, 'snapshot_sha256': sha256_file(destination / 'prerun_snapshot.json'), 'run': str(destination), 'started_utc': utc_now()}, exploratory=exploratory)
    write_json(destination / 'status.json', {'status': 'running', 'mode': mode, 'started_utc': utc_now()})
    try:
        first_fit = select_records(manifest, 'model_fit')[0]
        warm_image = preprocess_image(verified_path(config.data_root, first_fit['path'], first_fit['sha256']))
        results = {'mode': mode, 'snapshot_sha256': sha256_file(destination / 'prerun_snapshot.json'), 'models': {}}
        for model_name, model in models.items():
            for _ in range(EVALUATION_PROTOCOL['latency']['warmup']):
                model.score(warm_image)
            records = []
            maps_dir = destination / model_name / 'maps'
            maps_dir.mkdir(parents=True)
            # Incremental per-image log survives failure; never presented as a completed result.
            with (destination / model_name / 'predictions.jsonl').open('w') as stream:
                for index, row in enumerate(test):
                    path = verified_path(config.data_root, row['path'], row['sha256'])
                    info = validate_image(path)
                    if (info['width'], info['height']) != (row['width'], row['height']):
                        raise ValidationError('Image dimensions differ from frozen manifest')
                    mask = np.zeros((row['height'], row['width']), dtype=bool)
                    if row['is_anomaly']:
                        expected = f"{config.category}/ground_truth/{row['label']}/{Path(row['path']).stem}_mask.png"
                        if row['mask_path'] != expected:
                            raise ValidationError('Mask path does not match defect image')
                        mask_path = verified_path(config.data_root, row['mask_path'], row['mask_sha256'])
                        validate_image(mask_path, mask=True)
                        with Image.open(mask_path) as source:
                            mask = np.asarray(source.convert('L')) > 0
                        if mask.shape != (row['height'], row['width']):
                            raise ValidationError('Mask/image dimensions differ')
                    begin = perf_counter_ns()
                    image = preprocess_image(path)
                    model_start = perf_counter_ns()
                    score, small_map = model.score(image)
                    end = perf_counter_ns()
                    if not math.isfinite(score) or small_map.shape != (256, 256) or not np.isfinite(small_map).all():
                        raise ValidationError('Model returned invalid score or map; evaluation aborted')
                    full_map = np.asarray(Image.fromarray(small_map).resize((row['width'], row['height']), Image.Resampling.BILINEAR), dtype=np.float32)
                    predicted = bool(score > model.threshold) if math.isfinite(model.threshold) else None
                    outcome = 'uncalibrated' if predicted is None else ('TP' if predicted else 'FN') if row['is_anomaly'] else ('FP' if predicted else 'TN')
                    map_path = maps_dir / f'{index:04}.npy'
                    np.save(map_path, small_map, allow_pickle=False)
                    pixels = auroc(mask.ravel(), full_map.ravel())
                    record = {'path': row['path'], 'label': row['label'], 'is_anomaly': row['is_anomaly'],
                              'score': score, 'prediction': predicted, 'outcome': outcome,
                              'width': row['width'], 'height': row['height'], 'mask_path': row['mask_path'],
                              'map_path': map_path.relative_to(destination).as_posix(),
                              'pixel_auroc': pixels, 'positive_pixels': int(mask.sum()), 'negative_pixels': int((~mask).sum()),
                              'defect_region_mean': float(full_map[mask].mean()) if mask.any() else None,
                              'background_region_mean': float(full_map[~mask].mean()) if (~mask).any() else None,
                              'peak_inside_defect': bool(mask.ravel()[int(full_map.argmax())]) if mask.any() else None,
                              'model_ms': (end - model_start) / 1e6, 'pipeline_ms': (end - begin) / 1e6}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
            threshold = model.threshold if math.isfinite(model.threshold) else None
            results['models'][model_name] = {'threshold': threshold, 'calibration_count': model.metadata['calibration']['count'],
                                            'records': records, 'summary': summarize(records, threshold)}
        # Detect external changes to saved models/manifest during a run.
        if sha256_file(manifest_path) != snapshot['manifest_sha256'] or artifact_hashes(baseline_path) != frozen['baseline']['files'] or artifact_hashes(feature_path) != frozen['patchcore_style']['files']:
            raise ValidationError('Frozen inputs changed during evaluation')
        write_json(destination / 'results.json', results)
        create_report(destination, results, snapshot, config.data_root)
        write_json(destination / 'status.json', {'status': 'complete', 'mode': mode, 'finished_utc': utc_now()})
        return {'output': str(destination), 'mode': mode, 'status': 'complete', 'report': str(destination / 'REPORT.md')}
    except Exception as exc:
        write_json(destination / 'status.json', {'status': 'failed', 'mode': mode, 'reason': str(exc), 'failed_utc': utc_now(), 'note': 'Partial outputs are not a completed evaluation; official attempt remains recorded'})
        raise
