from pathlib import Path
import pytest
from edge_inspect.demo import sample,inspect_upload,discover,load_model
from edge_inspect.config import ValidationError

def test_upload_errors_and_actual_score():
    root=Path(__file__).resolve().parents[1]
    models=discover(root/'artifacts')
    baselines=[p for p,m in models if m['method'].startswith('rgb')]
    if not baselines: pytest.skip('Requires existing fixture baseline')
    model=load_model(baselines[0])
    for data,name in [(b'bad','x.png'),(sample(),'x.exe'),(b'x'*(10*1024*1024+1),'x.png')]:
        with pytest.raises(ValidationError):inspect_upload(model,data,name)
    result=inspect_upload(model,sample(),'sample.png')
    assert result['score']>=0 and result['overlay'].shape==result['heat'].shape

def test_streamlit_smoke():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    assert not app.exception
    if app.checkbox:
        app.checkbox[0].check().run()
        app.button[0].click().run(timeout=30)
        assert not app.exception and len(app.metric)==4


def test_no_models_setup_state(monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr('edge_inspect.demo.discover',lambda root: [])
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    assert not app.exception
    assert any('No model artifacts found' in item.value for item in app.warning)
    assert not app.metric
