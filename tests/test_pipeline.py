"""Tiny generated images validate software only, never model quality."""
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from edge_inspect.cli import main
from edge_inspect.config import Config, ValidationError
from edge_inspect.data import inspect_data, validate_image
from edge_inspect.manifest import build_manifest, manifest_bytes, select_records, write_manifest
from edge_inspect.preprocessing import iter_samples, preprocess_image, preprocess_mask
from edge_inspect.protocol import normal_threshold


@pytest.fixture
def dataset(tmp_path):
    config = Config(tmp_path / 'data', 'bottle', 42, tmp_path / 'artifacts', 'cpu')
    root = config.data_root / 'bottle'
    for split in ['train/good', 'test/good', 'test/crack', 'ground_truth/crack']:
        (root / split).mkdir(parents=True)
    for i in range(10):
        Image.new('RGB', (12, 10), (i * 20, 0, 10)).save(root / f'train/good/{i:03}.png')
    Image.new('RGB', (12, 10), (0, 80, 10)).save(root / 'test/good/000.png')
    Image.new('RGB', (12, 10), (0, 100, 10)).save(root / 'test/crack/000.png')
    mask = Image.new('L', (12, 10), 0)
    mask.putpixel((3, 3), 255)
    mask.save(root / 'ground_truth/crack/000_mask.png')
    return config


def test_manifest_reproducible_and_disjoint(dataset):
    first = build_manifest(dataset)
    assert manifest_bytes(first) == manifest_bytes(build_manifest(dataset))
    fit = select_records(first, 'model_fit')
    cal = select_records(first, 'normal_calibration')
    assert len(fit) == 8 and len(cal) == 2
    assert {r['path'] for r in fit}.isdisjoint(r['path'] for r in cal)
    assert all(r['official_split'] == 'train' for r in fit + cal)
    assert all(r['role'] == 'final_test' for r in first['records'] if r['official_split'] == 'test')
    for row in first['records']:
        assert row['sha256'] == hashlib.sha256((dataset.data_root / row['path']).read_bytes()).hexdigest()
        if row['mask_path']:
            assert row['mask_sha256'] == hashlib.sha256((dataset.data_root / row['mask_path']).read_bytes()).hexdigest()
    changed = build_manifest(replace(dataset, seed=123))
    assert {r['path'] for r in cal} != {r['path'] for r in select_records(changed, 'normal_calibration')}


def test_test_content_cannot_change_training_split(dataset):
    before = build_manifest(dataset)
    Image.new('RGB', (12, 10), (44, 22, 33)).save(dataset.data_root / 'bottle/test/crack/000.png')
    after = build_manifest(dataset)
    for role in ['model_fit', 'normal_calibration']:
        assert select_records(before, role) == select_records(after, role)


def test_manifest_snapshot_idempotent_and_raw_unchanged(dataset):
    files = {p: p.read_bytes() for p in dataset.data_root.rglob('*.png')}
    result = write_manifest(dataset)
    assert result == write_manifest(dataset)
    from pathlib import Path
    contents = Path(result['manifest']).read_bytes()
    assert hashlib.sha256(contents).hexdigest() == result['sha256']
    assert json.loads(contents)['protocol']['target_fpr'] == 0.05
    assert all(p.read_bytes() == content for p, content in files.items())


@pytest.mark.parametrize('problem', ['missing', 'corrupt', 'size', 'nonbinary', 'empty', 'orphan', 'rgb'])
def test_bad_masks_fail_without_artifacts(dataset, problem):
    path = dataset.data_root / 'bottle/ground_truth/crack/000_mask.png'
    if problem == 'missing':
        path.unlink()
    elif problem == 'corrupt':
        path.write_bytes(b'corrupt')
    elif problem == 'size':
        Image.new('L', (4, 4), 255).save(path)
    elif problem == 'nonbinary':
        Image.new('L', (12, 10), 127).save(path)
    elif problem == 'empty':
        Image.new('L', (12, 10), 0).save(path)
    elif problem == 'rgb':
        Image.new('RGB', (12, 10), 'white').save(path)
    else:
        Image.new('L', (12, 10), 255).save(path.with_name('orphan_mask.png'))
    with pytest.raises(ValidationError):
        write_manifest(dataset)
    assert not dataset.artifact_root.exists()


def test_duplicate_content_rejected(dataset):
    source = dataset.data_root / 'bottle/train/good/000.png'
    (dataset.data_root / 'bottle/test/good/000.png').write_bytes(source.read_bytes())
    with pytest.raises(ValidationError, match='Duplicate'):
        build_manifest(dataset)


def test_requires_two_training_images(dataset):
    for p in (dataset.data_root / 'bottle/train/good').glob('*.png'):
        if p.stem != '000':
            p.unlink()
    with pytest.raises(ValidationError, match='At least two'):
        build_manifest(dataset)


def test_final_test_guard_and_tampering(dataset):
    manifest = build_manifest(dataset)
    with pytest.raises(ValidationError, match='Final test loading'):
        list(iter_samples(manifest, dataset.data_root, 'final_test'))
    assert len(list(iter_samples(manifest, dataset.data_root, 'final_test', allow_final_test=True))) == 2
    tampered = deepcopy(manifest)
    row = next(r for r in tampered['records'] if r['official_split'] == 'test')
    row['role'] = 'model_fit'
    with pytest.raises(ValidationError, match='split/role'):
        list(iter_samples(tampered, dataset.data_root))
    row['official_split'] = 'train'
    row['label'] = 'good'
    row['is_anomaly'] = False
    row['mask_path'] = None
    with pytest.raises(ValidationError, match='Path disagrees'):
        list(iter_samples(tampered, dataset.data_root))


def test_modified_input_hash_rejected(dataset):
    manifest = build_manifest(dataset)
    row = select_records(manifest, 'model_fit')[0]
    Image.new('RGB', (12, 10), 'blue').save(dataset.data_root / row['path'])
    with pytest.raises(ValidationError, match='changed since manifest'):
        list(iter_samples(manifest, dataset.data_root))


def test_path_traversal_rejected(dataset):
    manifest = build_manifest(dataset)
    manifest['records'][0]['path'] = '../outside.png'
    with pytest.raises(ValidationError, match='Path disagrees'):
        list(iter_samples(manifest, dataset.data_root))


def test_preprocessing_exact_values_and_determinism(tmp_path):
    path = tmp_path / 'synthetic.png'
    Image.new('RGB', (3, 5), (255, 0, 0)).save(path)
    pixels = preprocess_image(path)
    assert pixels.shape == (3, 256, 256) and pixels.dtype == np.float32
    assert pixels.flags.c_contiguous
    np.testing.assert_array_equal(pixels, preprocess_image(path))
    assert np.all(pixels[0] == 1) and np.all(pixels[1:] == 0)
    expected = (np.array([1, 0, 0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(preprocess_image(path, pretrained=True)[:, 0, 0], expected, rtol=1e-6)
    gray = tmp_path / 'gray.png'
    Image.new('L', (2, 2), 128).save(gray)
    np.testing.assert_allclose(preprocess_image(gray), 128 / 255)


def test_nearest_mask_preserves_binary_edges(tmp_path):
    path = tmp_path / 'synthetic_mask.png'
    image = Image.new('L', (2, 2), 0)
    image.putpixel((0, 0), 255)
    image.save(path)
    mask = preprocess_mask(path)
    assert mask.shape == (256, 256) and mask.dtype == np.bool_
    assert mask[:128, :128].all() and not mask[128:, :].any() and not mask[:, 128:].any()


def test_corrupt_and_alpha_images_rejected(tmp_path):
    path = tmp_path / 'synthetic.png'
    path.write_bytes(b'bad image')
    with pytest.raises(ValidationError, match='Cannot decode'):
        preprocess_image(path)
    Image.new('RGBA', (3, 3)).save(path)
    with pytest.raises(ValidationError, match='mode'):
        validate_image(path)


def test_threshold_order_statistic_ties_and_small_sample():
    assert math.isinf(normal_threshold([1, 2]))
    assert normal_threshold(list(range(19))) == 18
    assert normal_threshold(list(range(42))) == 40
    assert normal_threshold([3] * 42) == 3
    assert not 3 > normal_threshold([3] * 42)


@pytest.mark.parametrize('scores', [[], [float('nan')], [float('inf')], [True]])
def test_invalid_scores(scores):
    with pytest.raises(ValidationError):
        normal_threshold(scores)


@pytest.mark.parametrize('alpha', [0, 1, -1, True, float('nan')])
def test_invalid_fpr(alpha):
    with pytest.raises(ValidationError):
        normal_threshold([1, 2], alpha)


def test_prepare_cli(dataset, tmp_path, capsys):
    config = tmp_path / 'config.toml'
    config.write_text('data_root = "data"\ncategory = "bottle"\nseed = 42\nartifact_root = "artifacts"\ndevice = "cpu"\n')
    assert main(['prepare-data', '--config', str(config)]) == 0
    output, logs = capsys.readouterr()
    assert json.loads(output)['counts'] == {'model_fit': 8, 'normal_calibration': 2, 'final_test': 2}
    assert json.loads(logs)['level'] == 'INFO'


def test_incomplete_test_set_rejected(dataset):
    (dataset.data_root / 'bottle/test/good/000.png').unlink()
    with pytest.raises(ValidationError, match='Complete category'):
        build_manifest(dataset)


def test_changed_protocol_rejected(dataset):
    manifest = deepcopy(build_manifest(dataset))
    manifest['protocol']['image_size'] = [32, 32]
    with pytest.raises(ValidationError, match='protocol'):
        list(iter_samples(manifest, dataset.data_root))


def test_changed_mask_hash_rejected(dataset):
    manifest = build_manifest(dataset)
    Image.new('L', (12, 10), 255).save(dataset.data_root / 'bottle/ground_truth/crack/000_mask.png')
    with pytest.raises(ValidationError, match='changed since manifest'):
        list(iter_samples(manifest, dataset.data_root, 'final_test', allow_final_test=True))
