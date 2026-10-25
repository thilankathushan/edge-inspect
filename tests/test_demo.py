from pathlib import Path
import pytest
from edge_inspect.demo import sample,inspect_upload,discover,load_model
from edge_inspect.config import ValidationError

@pytest.fixture
def demo_artifact(tmp_path):
    """Small generated software fixtures; never dataset or weight downloads."""
    from PIL import Image
    from edge_inspect.config import Config
    from edge_inspect.manifest import write_manifest
    from edge_inspect.baseline import train
    config=Config(tmp_path/'data','bottle',42,tmp_path/'artifacts','cpu')
    category=config.data_root/'bottle'
    for part in ['train/good','test/good','test/crack','ground_truth/crack']:
        (category/part).mkdir(parents=True)
    for i in range(100):
        Image.new('RGB',(8,10),(i,100,100)).save(category/f'train/good/{i:03}.png')
    Image.new('RGB',(8,10),'white').save(category/'test/good/000.png')
    Image.new('RGB',(8,10),'red').save(category/'test/crack/000.png')
    Image.new('L',(8,10),255).save(category/'ground_truth/crack/000_mask.png')
    manifest=Path(write_manifest(config)['manifest'])
    return Path(train(config,manifest,'demo','synthetic_fixture')['model'])


def test_upload_errors_and_actual_score(demo_artifact):
    model=load_model(demo_artifact)
    for data,name in [(b'bad','x.png'),(sample(),'x.exe'),(b'x'*(10*1024*1024+1),'x.png')]:
        with pytest.raises(ValidationError):inspect_upload(model,data,name)
    result=inspect_upload(model,sample(),'sample.png')
    assert result['score']>=0 and result['overlay'].shape==result['heat'].shape

def test_streamlit_smoke(demo_artifact,monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from edge_inspect.baseline import read_json
    # UI labels expect a path beneath the project artifacts directory.
    project=Path(__file__).resolve().parents[1]
    display_path=project/'artifacts/models/test-fixture'
    monkeypatch.setattr('edge_inspect.demo.discover',lambda root:[(display_path,read_json(demo_artifact/'metadata.json'))])
    monkeypatch.setattr('edge_inspect.demo.load_model',lambda path:load_model(demo_artifact))
    app=AppTest.from_file(str(project/'app.py')).run()
    assert not app.exception
    app.checkbox[0].check().run()
    app.button[0].click().run(timeout=30)
    assert not app.exception and len(app.metric)==4
    assert any('Fixture-trained' in item.value for item in app.warning)


def test_no_models_setup_state(monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr('edge_inspect.demo.discover',lambda root: [])
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    assert not app.exception
    assert any('No model artifacts found' in item.value for item in app.warning)
    assert not app.metric


def test_real_model_discovery_precedes_fixtures(tmp_path):
    import json
    entries=[('a-fixture','synthetic_fixture'),('z-real','official_mvtec_ad')]
    for name,kind in entries:
        target=tmp_path/'models'/name
        target.mkdir(parents=True)
        (target/'metadata.json').write_text(json.dumps({'method':'rgb-mean-template-v1','data_kind':kind}))
    assert [meta['data_kind'] for _,meta in discover(tmp_path)]==['official_mvtec_ad','synthetic_fixture']
