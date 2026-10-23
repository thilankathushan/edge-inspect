"""Synthetic software fixtures only; no real detector benchmarks."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from edge_inspect.baseline import TemplateModel, train
from edge_inspect.config import Config, ValidationError
from edge_inspect.manifest import write_manifest, select_records
from edge_inspect.baseline import read_json
from edge_inspect.preprocessing import preprocess_image
from edge_inspect.prediction import predict
from edge_inspect.cli import main


@pytest.fixture
def run(tmp_path):
    config = Config(tmp_path / 'data', 'bottle', 42, tmp_path / 'artifacts', 'cpu')
    category = config.data_root / 'bottle'
    for split in ['train/good', 'test/good', 'test/crack', 'ground_truth/crack']:
        (category / split).mkdir(parents=True)
    for i in range(100):
        Image.new('RGB', (8, 10), (i, 100, 100)).save(category / f'train/good/{i:03}.png')
    Image.new('RGB', (8, 10), 'white').save(category / 'test/good/000.png')
    Image.new('RGB', (8, 10), 'red').save(category / 'test/crack/000.png')
    Image.new('L', (8, 10), 255).save(category / 'ground_truth/crack/000_mask.png')
    manifest_path = Path(write_manifest(config)['manifest'])
    return config, manifest_path


def test_roundtrip_determinism_and_fit_only_statistics(run):
    config, manifest = run
    result = train(config, manifest, 'one', 'synthetic_fixture')
    model = TemplateModel.load(Path(result['model']))
    rows = select_records(read_json(manifest), 'model_fit')
    expected = np.mean([preprocess_image(config.data_root / r['path']).astype(np.float64) for r in rows], axis=0).astype(np.float32)
    np.testing.assert_array_equal(model.template, expected)
    image = preprocess_image(config.data_root / rows[0]['path'])
    score, anomaly_map = model.score(image)
    model.save(config.artifact_root / 'copy')
    loaded = TemplateModel.load(config.artifact_root / 'copy')
    score2, map2 = loaded.score(image)
    assert score == pytest.approx(score2, abs=1e-8)
    np.testing.assert_allclose(anomaly_map, map2, atol=1e-8)
    assert anomaly_map.shape == (256, 256) and score == float(anomaly_map.max())
    assert np.isfinite(anomaly_map).all() and (anomaly_map >= 0).all()
    again = train(config, manifest, 'two', 'synthetic_fixture')
    np.testing.assert_array_equal(model.template, TemplateModel.load(Path(again['model'])).template)
    assert again['threshold'] == result['threshold']
    assert result['calibration_count'] == 20 and result['status'] == 'calibrated'


def test_training_does_not_open_test_files(run):
    config, manifest = run
    for p in (config.data_root / 'bottle/test').rglob('*.png'):
        p.unlink()
    for p in (config.data_root / 'bottle/ground_truth').rglob('*.png'):
        p.unlink()
    assert train(config, manifest, 'no-test', 'synthetic_fixture')['fit_count'] == 80


def test_prediction_outputs_and_test_guard(run, tmp_path):
    config, manifest = run
    model = Path(train(config, manifest, 'model', 'synthetic_fixture')['model'])
    image_path = tmp_path / 'user.png'
    Image.new('RGB', (41, 29), (0, 0, 255)).save(image_path)
    result = predict(config, model, image_path, 'prediction')
    assert result['is_anomaly'] is True
    output = Path(result['output'])
    assert np.load(output / 'anomaly_map.npy').shape == (256, 256)
    assert np.load(output / 'anomaly_map_original.npy').shape == (29, 41)
    assert json.loads((output / 'prediction.json').read_text())['model_data_kind'] == 'synthetic_fixture'
    with Image.open(output / 'overlay.png') as overlay:
        assert overlay.height == 256 + 90
    repeated = predict(config, model, image_path, 'prediction2')
    assert repeated['score'] == result['score']
    with pytest.raises(ValidationError, match='already exists'):
        predict(config, model, image_path, 'prediction')
    test_image = config.data_root / 'bottle/test/crack/000.png'
    with pytest.raises(ValidationError, match='reserved'):
        predict(config, model, test_image, 'forbidden')
    copied = tmp_path / 'copied.png'
    copied.write_bytes(test_image.read_bytes())
    with pytest.raises(ValidationError, match='reserved'):
        predict(config, model, copied, 'forbidden-copy')


def test_calibration_cannot_change_template(run):
    config, manifest_path = run
    one = TemplateModel.load(Path(train(config, manifest_path, 'before', 'synthetic_fixture')['model']))
    manifest = read_json(manifest_path)
    calibration = select_records(manifest, 'normal_calibration')
    for i, row in enumerate(calibration):
        Image.new('RGB', (8, 10), (150 + i, 200, 200)).save(config.data_root / row['path'])
    new_manifest = Path(write_manifest(config)['manifest'])
    two = TemplateModel.load(Path(train(config, new_manifest, 'after', 'synthetic_fixture')['model']))
    np.testing.assert_array_equal(one.template, two.template)
    assert one.threshold != two.threshold


def test_small_calibration_is_uncalibrated(run, tmp_path):
    config, manifest = run
    for p in (config.data_root / 'bottle/train/good').glob('*.png'):
        if int(p.stem) >= 10:
            p.unlink()
    manifest = Path(write_manifest(config)['manifest'])
    trained = train(config, manifest, 'small', 'synthetic_fixture')
    assert trained['threshold'] is None and trained['status'] == 'insufficient_calibration'
    user = tmp_path / 'user.png'
    Image.new('RGB', (20, 20), 'blue').save(user)
    result = predict(config, Path(trained['model']), user, 'small-prediction')
    assert result['is_anomaly'] is None and result['decision'] == 'uncalibrated'


def test_tampered_artifact_rejected(run):
    config, manifest = run
    model = Path(train(config, manifest, 'tamper', 'synthetic_fixture')['model'])
    (model / 'template.npy').write_bytes(b'broken')
    with pytest.raises(ValidationError, match='hash'):
        TemplateModel.load(model)


@pytest.mark.parametrize('kind', ['corrupt', 'too_small', 'too_large', 'format', 'alpha'])
def test_bad_prediction_images(run, tmp_path, kind):
    config, manifest = run
    model = Path(train(config, manifest, 'model', 'synthetic_fixture')['model'])
    path = tmp_path / 'user.png'
    if kind == 'corrupt':
        path.write_bytes(b'bad')
    elif kind == 'too_small':
        Image.new('RGB', (1, 4)).save(path)
    elif kind == 'too_large':
        Image.new('RGB', (8193, 2)).save(path)
    elif kind == 'format':
        Image.new('RGB', (8, 8)).save(path, format='GIF')
    else:
        Image.new('RGBA', (8, 8)).save(path)
    with pytest.raises(ValidationError):
        predict(config, model, path, 'bad-input')
    assert not (config.artifact_root / 'predictions').exists()


def test_cli_end_to_end(run, tmp_path, capsys):
    config, manifest = run
    config_path = tmp_path / 'config.toml'
    config_path.write_text('data_root="data"\ncategory="bottle"\nseed=42\nartifact_root="artifacts"\ndevice="cpu"\n')
    assert main(['train', '--config', str(config_path), '--manifest', str(manifest), '--run-name', 'cli', '--data-kind', 'synthetic_fixture']) == 0
    result = json.loads(capsys.readouterr().out)
    user = tmp_path / 'user.png'
    Image.new('RGB', (10, 20), 'blue').save(user)
    assert main(['predict', '--config', str(config_path), '--model', result['model'], '--image', str(user), '--output-name', 'cli-prediction']) == 0
    assert json.loads(capsys.readouterr().out)['decision'] == 'anomaly'


def test_stored_threshold_tampering_rejected(run):
    config, manifest = run
    folder = Path(train(config, manifest, 'threshold', 'synthetic_fixture')['model'])
    metadata = read_json(folder / 'metadata.json')
    metadata['calibration']['threshold'] = 0.99999
    (folder / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match='threshold'):
        TemplateModel.load(folder)


def test_invalid_manifest_cli_returns_error(run, tmp_path, capsys):
    config, manifest = run
    config_file = tmp_path / 'config.toml'
    config_file.write_text('data_root="data"\ncategory="bottle"\nseed=42\nartifact_root="artifacts"\ndevice="cpu"\n')
    bad = tmp_path / 'bad.json'
    bad.write_text('{broken')
    assert main(['train', '--config', str(config_file), '--manifest', str(bad), '--run-name', 'bad', '--data-kind', 'synthetic_fixture']) == 2
    assert 'Cannot read JSON' in capsys.readouterr().err
