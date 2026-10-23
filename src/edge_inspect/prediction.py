"""Single-image predictions and explicitly labeled residual visualizations."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .baseline import TemplateModel, output_directory, write_json, read_json
from .config import Config, ValidationError
from .data import sha256_file, validate_image
from .preprocessing import preprocess_image


def predict(config: Config, model_path: Path, image_path: Path, output_name: str, *, onnx_path: Path | None = None) -> dict:
    method = read_json(model_path / 'metadata.json').get('method')
    feature_model = method == 'patchcore-style-resnet18-v1'
    if feature_model:
        from .patchcore import PatchModel
        model = PatchModel.load(model_path, onnx_path=onnx_path)
    else:
        if onnx_path is not None:
            raise ValidationError('ONNX is available only for the feature extractor')
        model = TemplateModel.load(model_path)
    if model.metadata['category'] != config.category:
        raise ValidationError('Model category differs from configuration')
    image_path = image_path.expanduser().resolve()
    test_root = (config.data_root / config.category / 'test').resolve()
    if image_path.is_relative_to(test_root):
        raise ValidationError('Official test inputs are reserved for final evaluation')
    info = validate_image(image_path)
    digest = sha256_file(image_path)
    if digest in model.metadata.get('reserved_test_sha256', []):
        raise ValidationError('Image hash matches reserved final-test data')
    destination = output_directory(config.artifact_root / 'predictions', output_name)
    score, anomaly_map = model.score(preprocess_image(image_path))
    finite = math.isfinite(model.threshold)
    display_scale = max(model.threshold, 1e-6) if feature_model and finite else (1.0 if not feature_model else max((r['score'] for r in model.metadata['calibration']['scores']), default=1.0))
    display_scale = max(display_scale, 1e-6)
    map_label = 'PatchCore-style distance' if feature_model else 'RGB residual'
    result = {
        'runtime': 'onnx-extractor-plus-torch-scoring' if onnx_path else 'original',
        'onnx_sha256': sha256_file(onnx_path / 'extractor.onnx') if onnx_path else None,
        'method': model.metadata['method'], 'model_data_kind': model.metadata['data_kind'],
        'input_sha256': digest, 'input_dimensions': [info['width'], info['height']],
        'score': score, 'threshold': model.threshold if finite else None,
        'is_anomaly': bool(score > model.threshold) if finite else None,
        'decision': ('anomaly' if score > model.threshold else 'within_threshold') if finite else 'uncalibrated',
        'calibration_status': model.metadata['calibration']['status'],
        'map_shape': list(anomaly_map.shape), 'original_map_shape': [info['height'], info['width']],
        'overlay_note': f'{map_label}, not defect probability; sqrt display; clipped at saved calibration scale {display_scale:.6g}',
        'overlay_scale': display_scale,
        'model_metadata_sha256': sha256_file(model_path / 'metadata.json'),
    }
    original_map = np.asarray(Image.fromarray(anomaly_map).resize((info['width'], info['height']), Image.Resampling.BILINEAR), dtype=np.float32)
    # Preview uses bounded geometry; machine-readable map retains original geometry.
    with Image.open(image_path) as source:
        preview = source.convert('RGB')
        scale = min(768 / max(preview.size), max(1, 256 / min(preview.size)))
        preview = preview.resize(tuple(max(1, round(d * scale)) for d in preview.size), Image.Resampling.BILINEAR)
    display_map = np.asarray(Image.fromarray(anomaly_map).resize(preview.size, Image.Resampling.BILINEAR))
    alpha = (0.65 * np.sqrt(np.clip(display_map / display_scale, 0, 1)))[..., None]
    overlay = ((1 - alpha) * np.asarray(preview) + alpha * np.array([255, 80, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (max(640, preview.width), preview.height + 90), 'white')
    canvas.paste(Image.fromarray(overlay), ((canvas.width - preview.width) // 2, 90))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f'EDGE INSPECT | {map_label} (not defect probability)', fill='black')
    draw.text((8, 25), f"Score {score:.6g} | {result['decision']} | model data: {result['model_data_kind']}", fill='black')
    draw.text((8, 44), f'Orange: 0..{display_scale:.5g}, sqrt display scale; no pixel threshold', fill='black')
    draw.text((8, 63), 'Synthetic fixture model: not validated on real defects' if model.metadata['data_kind'] == 'synthetic_fixture' else 'No official test evaluation performed', fill='black')
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / 'anomaly_map.npy', anomaly_map, allow_pickle=False)
    np.save(destination / 'anomaly_map_original.npy', original_map, allow_pickle=False)
    canvas.save(destination / 'overlay.png')
    write_json(destination / 'prediction.json', result)
    return {**result, 'output': str(destination)}
