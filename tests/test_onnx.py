"""ONNX checks use synthetic fixtures and explicit cached weights, never downloads."""
from pathlib import Path
import numpy as np
import pytest
from edge_inspect.config import ValidationError
from edge_inspect.onnx_backend import normalize
from edge_inspect.runtime_benchmark import parity_record, benchmark


def test_normalization_and_validation():
    from edge_inspect.features import FEATURE_CONFIG
    image=np.full((3,256,256),.5,dtype=np.float32)
    result=normalize(image)
    assert result.shape==(1,3,256,256) and result.dtype==np.float32
    for c in range(3):
        assert result[0,c,0,0]==pytest.approx((.5-FEATURE_CONFIG['preprocessing']['mean'][c])/FEATURE_CONFIG['preprocessing']['std'][c])
    with pytest.raises(ValidationError): normalize(np.zeros((3,10,10),dtype=np.float32))
    with pytest.raises(ValidationError): normalize(np.full((3,256,256),np.nan,dtype=np.float32))


def test_numerical_tolerance_never_hides_decision_flip():
    z=np.zeros((2,2),dtype=np.float32)
    result=parity_record(1.,1.000001,z,z,z,z,1.)
    assert result['score_allclose'] and result['decision_changed']
    assert result['original_decision'] is False and result['onnx_decision'] is True
    assert not parity_record(1.,2.,z,z,z,z,float('inf'))['decision_changed']


def test_invalid_benchmark_parameters():
    with pytest.raises(ValidationError,match='repeats'):
        benchmark(None,None,None,None,None,repeats=1)


def test_actual_export_score_parity_prediction_and_benchmark(tmp_path):
    pytest.importorskip('onnxruntime')
    pytest.importorskip('onnxscript')
    pytest.importorskip('psutil')
    pytest.importorskip('torch')
    from PIL import Image
    from edge_inspect.config import Config
    from edge_inspect.manifest import write_manifest
    from edge_inspect.patchcore import train_features, PatchModel
    from edge_inspect.onnx_backend import export_onnx, OnnxExtractor
    from edge_inspect.baseline import read_json
    from edge_inspect.prediction import predict
    from edge_inspect.preprocessing import preprocess_image
    weights=Path(__file__).resolve().parents[1]/'artifacts/weights/resnet18-f37072fd.pth'
    if not weights.exists(): pytest.skip('Explicit cached pretrained weights required; tests never download')
    config=Config(tmp_path/'data','bottle',42,tmp_path/'outputs','cpu')
    root=config.data_root/'bottle'
    for part in ['train/good','test/good','test/fixture','ground_truth/fixture']:
        (root/part).mkdir(parents=True)
    for i in range(10): Image.new('RGB',(9,11),(i*20,100,100)).save(root/f'train/good/{i}.png')
    Image.new('RGB',(9,11),'white').save(root/'test/good/a.png')
    Image.new('RGB',(9,11),'red').save(root/'test/fixture/a.png')
    Image.new('L',(9,11),255).save(root/'ground_truth/fixture/a_mask.png')
    manifest=Path(write_manifest(config)['manifest'])
    model=Path(train_features(config,manifest,'source','synthetic_fixture',weights)['model'])
    exported=Path(export_onnx(config,model,'export')['output'])
    a,b=PatchModel.load(model),PatchModel.load(model,onnx_path=exported)
    user=tmp_path/'user.png'
    Image.new('RGB',(12,14),(95,105,107)).save(user)
    image=preprocess_image(user)
    aa,bb=a.extractor.embeddings(image),b.extractor.embeddings(image)
    sa,ma=a.score(image);sb,mb=b.score(image)
    check=parity_record(sa,sb,aa,bb,ma,mb,a.threshold)
    assert check['embedding_allclose'] and check['score_allclose'] and check['map_allclose']
    assert not check['decision_changed']
    assert ma.shape==mb.shape==(256,256)
    assert predict(config,model,user,'onnx-user',onnx_path=exported)['runtime']=='onnx-extractor-plus-torch-scoring'
    output=Path(benchmark(config,manifest,model,exported,'check',warmup=1,repeats=2)['output'])
    data=read_json(output/'results.json')
    assert data['numerical_parity_passed'] and data['decision_changes']==0
    for runtime in data['runtimes'].values():
        assert runtime['latency']['end_to_end']['count']==4
        assert runtime['memory']['process_lifetime_peak_rss_bytes']>0
        assert runtime['quality']['image_auroc']['value'] is None
    with pytest.raises(ValidationError,match='Output already exists'):
        export_onnx(config,model,'export')
    # Integrity and source binding are mandatory even when the graph is valid.
    metadata=(model/'metadata.json').read_text()
    (model/'metadata.json').write_text(metadata+'\n')
    with pytest.raises(ValidationError,match='different source'):
        OnnxExtractor(exported,model)
    (model/'metadata.json').write_text(metadata)
    with (exported/'extractor.onnx').open('ab') as stream: stream.write(b'corrupt')
    with pytest.raises(ValidationError,match='hash mismatch'):
        OnnxExtractor(exported,model)
