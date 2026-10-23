"""Software fixtures, including optional offline tests of official pretrained weights."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
import torch
from edge_inspect.baseline import read_json, training_records
from edge_inspect.config import Config, ValidationError
from edge_inspect.features import Extractor, FEATURE_CONFIG, WEIGHT_NAME, check_official_weights
from edge_inspect.manifest import write_manifest
from edge_inspect.patchcore import PatchModel, Reservoir, nearest_distances, train_features
from edge_inspect.preprocessing import preprocess_image
from edge_inspect.prediction import predict


@pytest.fixture(scope='module')
def weights():
    path = Path(__file__).resolve().parents[1] / 'artifacts/weights' / WEIGHT_NAME
    if not path.exists():
        pytest.skip('Offline integration requires explicit fetch-weights; tests never download')
    return path


@pytest.fixture(scope='module')
def extractor(weights):
    return Extractor(weights, official=True)


@pytest.fixture
def fixture_data(tmp_path):
    config = Config(tmp_path / 'data', 'bottle', 42, tmp_path / 'outputs', 'cpu')
    root = config.data_root / 'bottle'
    for part in ['train/good', 'test/good', 'test/fixture', 'ground_truth/fixture']:
        (root / part).mkdir(parents=True)
    for i in range(10):
        Image.new('RGB', (9, 11), (i * 20, 100, 100)).save(root / f'train/good/{i:03}.png')
    Image.new('RGB', (9, 11), 'white').save(root / 'test/good/000.png')
    Image.new('RGB', (9, 11), 'red').save(root / 'test/fixture/000.png')
    Image.new('L', (9, 11), 255).save(root / 'ground_truth/fixture/000_mask.png')
    manifest = Path(write_manifest(config)['manifest'])
    return config, manifest


def test_reservoir_bound_reproducibility_and_origins():
    patches = np.arange(12800, dtype=np.float32).reshape(100, 128)
    a, b, c = [Reservoir(13, 128, seed) for seed in [42, 42, 43]]
    for sampler in [a, b, c]:
        sampler.add(patches[:50], 'fit-a')
        sampler.add(patches[50:], 'fit-b')
    bank, origins = a.finish()
    assert bank.shape == (13, 128) and bank.nbytes == 13 * 128 * 4
    np.testing.assert_array_equal(bank, b.finish()[0])
    assert origins == b.finish()[1] and not np.array_equal(bank, c.finish()[0])
    for vector, origin in zip(bank, origins):
        offset = 0 if origin['path'] == 'fit-a' else 50
        np.testing.assert_array_equal(vector, patches[offset + origin['patch_index']])
    assert a.seen == 100


def test_chunked_nn_matches_bruteforce_and_chunk_bounds(monkeypatch):
    rng = np.random.default_rng(12)
    query = rng.normal(size=(67, 128)).astype(np.float32)
    bank = rng.normal(size=(519, 128)).astype(np.float32)
    expected = np.linalg.norm(query[:, None] - bank[None], axis=2).min(axis=1)
    original = torch.cdist
    shapes = []
    def observed(q, b, **kwargs):
        shapes.append((q.shape[1], b.shape[1]))
        return original(q, b, **kwargs)
    monkeypatch.setattr(torch, 'cdist', observed)
    np.testing.assert_allclose(nearest_distances(query, bank), expected, rtol=1e-6, atol=1e-6)
    assert len(shapes) == 4 and max(q for q, b in shapes) <= 64 and max(b for q, b in shapes) <= 512
    np.testing.assert_allclose(nearest_distances(bank[:2], bank), 0, atol=1e-6)


def test_missing_and_invalid_weights_do_not_download(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('Implicit network call'))
    with pytest.raises(ValidationError, match='fetch-weights'):
        check_official_weights(tmp_path / 'missing.pth')
    bad = tmp_path / 'bad.pth'
    bad.write_bytes(b'not weights')
    with pytest.raises(ValidationError, match='official'):
        check_official_weights(bad)


def test_preprocessing_matches_shared_path_and_eval_mode(extractor, tmp_path):
    path = tmp_path / 'fixture.png'
    Image.new('RGB', (19, 23), (100, 80, 60)).save(path)
    raw = preprocess_image(path)
    normalized = preprocess_image(path, pretrained=True)
    with torch.inference_mode():
        expected = extractor.network(torch.from_numpy(normalized)[None])['patches'][0].permute(1, 2, 0).reshape(-1, 128).numpy()
    np.testing.assert_allclose(extractor.embeddings(raw), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(extractor.embeddings(raw), extractor.embeddings(raw))
    assert not extractor.network.training
    assert all(not p.requires_grad for p in extractor.network.parameters())
    assert all(p.device.type == 'cpu' for p in extractor.network.parameters())


def test_train_save_load_leakage_and_prediction(fixture_data, weights, monkeypatch, tmp_path):
    config, manifest_path = fixture_data
    manifest = read_json(manifest_path)
    fit, calibration, _ = training_records(config, manifest)
    # Training must succeed with all test/mask bytes absent, and must never decode them.
    for path in (config.data_root / 'bottle/test').rglob('*.png'):
        path.unlink()
    for path in (config.data_root / 'bottle/ground_truth').rglob('*.png'):
        path.unlink()
    result = train_features(config, manifest_path, 'features', 'synthetic_fixture', weights)
    assert result['bank_rows'] == 4096 and result['bank_bytes'] == 2_097_152
    assert result['patches_seen'] == len(fit) * 1024
    assert result['status'] == 'insufficient_calibration'
    folder = Path(result['model'])
    model = PatchModel.load(folder)
    origins = {r['path'] for r in model.metadata['bank_origins']}
    assert origins <= {r['path'] for r in fit}
    assert not origins & {r['path'] for r in calibration}
    image_path = tmp_path / 'query.png'
    Image.new('RGB', (17, 21), 'blue').save(image_path)
    image = preprocess_image(image_path)
    score, anomaly_map = model.score(image)
    assert anomaly_map.shape == (256, 256) and np.isfinite(anomaly_map).all()
    assert score >= float(anomaly_map.max()) - 1e-6
    copy = config.artifact_root / 'copy'
    model.save(copy)
    other = PatchModel.load(copy)
    np.testing.assert_allclose(other.score(image)[1], anomaly_map, rtol=1e-6, atol=1e-6)
    assert other.score(image)[0] == pytest.approx(score, abs=1e-6)
    prediction = predict(config, copy, image_path, 'feature-prediction')
    assert prediction['method'] == 'patchcore-style-resnet18-v1'
    assert prediction['decision'] == 'uncalibrated'
    assert np.load(Path(prediction['output']) / 'anomaly_map_original.npy').shape == (21, 17)
    # Calibration byte changes must leave the sampled bank unchanged.
    from edge_inspect.data import sha256_file
    for i, row in enumerate(calibration):
        path = config.data_root / row['path']
        Image.new('RGB', (9, 11), (10, 150 + i, 20)).save(path)
        next(r for r in manifest['records'] if r['path'] == row['path'])['sha256'] = sha256_file(path)
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(manifest))
    second = train_features(config, changed, 'changed-calibration', 'synthetic_fixture', weights)
    np.testing.assert_array_equal(model.bank, PatchModel.load(Path(second['model'])).bank)
    metadata = read_json(folder / 'metadata.json')
    metadata['feature_config']['memory_bank_capacity'] = 8192
    (folder / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match='configuration'):
        PatchModel.load(folder)


def test_frozen_comparison_matches_implementation():
    from edge_inspect.protocol import PROTOCOL
    frozen = read_json(Path(__file__).resolve().parents[1] / 'configs/frozen_comparison.json')
    assert frozen['feature_model'] == FEATURE_CONFIG
    assert frozen['baseline']['method'] == 'rgb-mean-template-v1'
    assert frozen['data_protocol'] == PROTOCOL
