"""Synthetic software fixtures only; no dataset benchmarks."""
import json
from pathlib import Path

import pytest
from PIL import Image

from edge_inspect.cli import main
from edge_inspect.config import ValidationError, load_config
from edge_inspect.data import inspect_data


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('data_root = "data"\ncategory = "bottle"\nseed = 42\nartifact_root = "artifacts"\ndevice = "cpu"\n')
    return path


def test_paths_are_config_relative(config_file, monkeypatch, tmp_path):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(config_file)
    assert config.data_root == tmp_path / 'data'
    assert config.artifact_root == tmp_path / 'artifacts'
    assert not config.artifact_root.exists()


@pytest.mark.parametrize(('old', 'new'), [
    ('category = "bottle"', 'category = "../escape"'),
    ('category = "bottle"', 'category = 1'),
    ('seed = 42', 'seed = true'),
    ('seed = 42', 'seed = -1'),
    ('seed = 42', 'seed = 4294967296'),
    ('device = "cpu"', 'device = "mps"'),
    ('data_root = "data"', 'data_root = ""'),
    ('artifact_root = "artifacts"', 'artifact_root = "data/output"'),
    ('artifact_root = "artifacts"', 'artifact_root = "."'),
    ('seed = 42', 'sead = 42'),
    ('seed = 42', 'seed = [broken'),
])
def test_invalid_config(config_file, old, new):
    config_file.write_text(config_file.read_text().replace(old, new))
    with pytest.raises(ValidationError):
        load_config(config_file)


def test_missing_config(tmp_path):
    with pytest.raises(ValidationError, match='Cannot read config'):
        load_config(tmp_path / 'missing.toml')


def test_file_instead_of_directory(config_file):
    (config_file.parent / 'data').write_text('not a directory')
    with pytest.raises(ValidationError, match='directory'):
        load_config(config_file)


def make_image(config, relative):
    path = config.data_root / config.category / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('L' if relative.startswith('ground_truth/') else 'RGB', (8, 8),
              color=255 if relative.startswith('ground_truth/') else
              ('white' if relative.startswith('train/') else 'red')).save(path)
    return path


def test_inventory_reads_images_without_creating_artifacts(config_file):
    config = load_config(config_file)
    make_image(config, 'train/good/normal.png')
    make_image(config, 'test/crack/defect.png')
    make_image(config, 'ground_truth/crack/defect_mask.png')
    result = inspect_data(config)
    assert result['counts'] == {'train/good': 1, 'test/crack': 1, 'ground_truth/crack': 1}
    assert not config.artifact_root.exists()


def test_missing_data(config_file):
    with pytest.raises(ValidationError, match='Category directory not found'):
        inspect_data(load_config(config_file))


def test_empty_data(config_file):
    config = load_config(config_file)
    (config.data_root / config.category).mkdir(parents=True)
    with pytest.raises(ValidationError, match='No normal training images'):
        inspect_data(config)


@pytest.mark.parametrize('relative', ['train/crack/a.png', 'unexpected/a.png'])
def test_invalid_layout(config_file, relative):
    config = load_config(config_file)
    make_image(config, relative)
    with pytest.raises(ValidationError):
        inspect_data(config)


def test_corrupt_image(config_file):
    config = load_config(config_file)
    path = make_image(config, 'train/good/a.png')
    path.write_bytes(b'not an image')
    with pytest.raises(ValidationError, match='Cannot decode image'):
        inspect_data(config)


def test_symlink_rejected(config_file, tmp_path):
    config = load_config(config_file)
    path = make_image(config, 'train/good/a.png')
    (path.parent / 'alias.png').symlink_to(path)
    with pytest.raises(ValidationError, match='symlinks'):
        inspect_data(config)


def test_cli_json_and_logs(config_file, capsys):
    assert main(['check-config', '--config', str(config_file)]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)['device'] == 'cpu'
    assert json.loads(err)['level'] == 'INFO'


def test_cli_missing_data_is_actionable(config_file, capsys):
    assert main(['inspect-data', '--config', str(config_file)]) == 2
    out, err = capsys.readouterr()
    assert out == ''
    assert 'Set data_root/category' in json.loads(err)['event']
