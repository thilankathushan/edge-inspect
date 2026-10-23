"""Evaluation guards on synthetic software fixtures only."""
from pathlib import Path
import pytest
from edge_inspect.config import Config, ValidationError
from edge_inspect.evaluation import reserve_holdout, prepare_evaluation
from edge_inspect.reporting import representatives


def test_ledger_preserves_first_attempt(tmp_path):
    reserve_holdout(tmp_path,'abc',{'run':'first'})
    with pytest.raises(ValidationError,match='already has'):
        reserve_holdout(tmp_path,'abc',{'run':'second'})
    reserve_holdout(tmp_path,'abc',{'run':'exploratory'},exploratory=True)
    assert 'first' in (tmp_path/'heldout-ledger/abc.json').read_text()


def test_missing_data_fails_before_artifacts(tmp_path):
    config = Config(tmp_path/'data','bottle',42,tmp_path/'outputs','cpu')
    with pytest.raises(ValidationError,match='unavailable'):
        prepare_evaluation(config,*([tmp_path/'missing']*4),False)
    assert not config.artifact_root.exists()


def test_representatives_deterministic():
    rows = [{'outcome':'FN','score':i,'path':str(i)} for i in [3,1,2]]
    assert representatives(rows)['FN']['score'] == 2
    assert 'FP' not in representatives(rows)


def test_fixture_cannot_be_official():
    from edge_inspect.config import load_config
    root = Path(__file__).resolve().parents[1]
    fixture = root/'artifacts/fixture-evaluation-verified'
    if not fixture.exists(): pytest.skip('Run explicit fixture verification first; no downloads in tests')
    manifest = next((fixture/'outputs/manifests').glob('*.json'))
    with pytest.raises(ValidationError,match='provenance'):
        prepare_evaluation(load_config(fixture/'config.toml'),manifest,fixture/'outputs/models/baseline',fixture/'outputs/models/features',root/'configs/frozen_comparison.json',False)


def test_end_to_end_snapshot_and_undefined_threshold(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    pytest.importorskip('matplotlib')
    from PIL import Image
    from edge_inspect.baseline import train, TemplateModel, read_json
    from edge_inspect.patchcore import train_features
    from edge_inspect.manifest import write_manifest
    from edge_inspect.evaluation import evaluate
    project = Path(__file__).resolve().parents[1]
    weights = project/'artifacts/weights/resnet18-f37072fd.pth'
    if not weights.exists(): pytest.skip('Explicit offline weight download required')
    config = Config(tmp_path/'data','bottle',42,tmp_path/'outputs','cpu')
    category = config.data_root/'bottle'
    for part in ['train/good','test/good','test/fixture','ground_truth/fixture']:
        (category/part).mkdir(parents=True)
    for i in range(10):
        Image.new('RGB',(8,9),(i*10,100,100)).save(category/f'train/good/{i}.png')
    Image.new('RGB',(8,9),'white').save(category/'test/good/a.png')
    Image.new('RGB',(8,9),'red').save(category/'test/fixture/a.png')
    Image.new('L',(8,9),255).save(category/'ground_truth/fixture/a_mask.png')
    manifest = Path(write_manifest(config)['manifest'])
    baseline = Path(train(config,manifest,'baseline','synthetic_fixture')['model'])
    features = Path(train_features(config,manifest,'features','synthetic_fixture',weights)['model'])
    original = TemplateModel.score
    def checked_score(self,image):
        assert (config.artifact_root/'evaluations/check/prerun_snapshot.json').exists()
        return original(self,image)
    monkeypatch.setattr(TemplateModel,'score',checked_score)
    result = evaluate(config,manifest,baseline,features,project/'configs/frozen_comparison.json','check',fixture=True)
    out = Path(result['output'])
    data = read_json(out/'results.json')
    assert data['mode'] == 'synthetic_fixture_verification'
    for model in data['models'].values():
        assert model['summary']['operating']['tp'] is None
        assert model['summary']['pixel_macro']['auroc']['value'] is None
        assert all(r['outcome']=='uncalibrated' for r in model['records'])
    assert read_json(out/'status.json')['status'] == 'complete'
    assert (out/'comparison.png').stat().st_size > 0
    assert 'SYNTHETIC' in (out/'REPORT.md').read_text()
