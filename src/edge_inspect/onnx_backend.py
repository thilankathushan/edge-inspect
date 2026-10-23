"""ONNX backbone only. Pillow/NumPy preprocessing and torch 1-NN remain external."""
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .baseline import output_directory, read_json, write_json, validate_tensor
from .config import ValidationError
from .data import sha256_file
from .features import FEATURE_CONFIG

OPSET = 18
EXPORT_SPEC = {'opset': OPSET, 'input': 'normalized_rgb', 'input_shape': [1,3,256,256],
               'output': 'features', 'output_shape': [1,128,32,32], 'dtype': 'float32',
               'scope': 'ResNet18 through layer2 only; normalization, patch layout, nearest neighbors, map resize and threshold are external',
               'dynamic_shapes': False, 'quantized': False}
PARITY = {'embedding_atol': 1e-4, 'embedding_rtol': 1e-4, 'score_atol': 1e-4, 'score_rtol': 1e-4,
          'map_atol': 1e-4, 'map_rtol': 1e-4, 'decision': 'exact strict score > unchanged saved threshold'}


def normalize(image):
    validate_tensor(image)
    mean = np.asarray(FEATURE_CONFIG['preprocessing']['mean'],dtype=np.float32)[:,None,None]
    std = np.asarray(FEATURE_CONFIG['preprocessing']['std'],dtype=np.float32)[:,None,None]
    return np.ascontiguousarray(((image-mean)/std)[None])


def runtime_session(path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValidationError("Install the onnx extra in the project environment") from exc
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    try:
        return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    except Exception as exc:
        raise ValidationError(f'Cannot load ONNX extractor: {exc}') from exc


class OnnxExtractor:
    def __init__(self, directory: Path, source_model: Path):
        metadata = read_json(directory/'export.json')
        if metadata.get('spec') != EXPORT_SPEC or metadata.get('feature_config') != FEATURE_CONFIG:
            raise ValidationError('Unsupported ONNX export specification')
        if metadata.get('model_metadata_sha256') != sha256_file(source_model/'metadata.json') or metadata.get('source_extractor_sha256') != sha256_file(source_model/'extractor.pth'):
            raise ValidationError('ONNX export belongs to a different source model')
        if metadata.get('onnx_sha256') != sha256_file(directory/'extractor.onnx'):
            raise ValidationError('ONNX artifact hash mismatch')
        self.session = runtime_session(directory/'extractor.onnx')
        if self.session.get_inputs()[0].shape != EXPORT_SPEC['input_shape'] or self.session.get_outputs()[0].shape != EXPORT_SPEC['output_shape']:
            raise ValidationError('Unexpected ONNX geometry')

    def embeddings(self, image):
        features = self.session.run(['features'],{'normalized_rgb':normalize(image)})[0]
        if features.shape != (1,128,32,32) or not np.isfinite(features).all():
            raise ValidationError('Invalid ONNX feature output')
        return np.ascontiguousarray(features[0].transpose(1,2,0).reshape(-1,128))


def export_onnx(config, model_path, run_name):
    try:
        import torch
        import onnx
        import onnxscript  # Explicit prerequisite for the dynamo exporter.
    except ImportError as exc:
        raise ValidationError("Install export dependencies with pip install -e '.[features,onnx]'") from exc
    from .patchcore import PatchModel
    model = PatchModel.load(model_path)
    if model.metadata['category'] != config.category:
        raise ValidationError('Model category differs from configuration')
    destination = output_directory(config.artifact_root/'exports',run_name)
    destination.mkdir(parents=True)
    class Backbone(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network=network
        def forward(self, normalized_rgb):
            return self.network(normalized_rgb)['patches']
    # Fixed generated input for export mechanics, not calibration or benchmark data.
    dummy = torch.zeros((1,3,256,256),dtype=torch.float32)
    try:
        torch.onnx.export(Backbone(model.extractor.network).eval(),(dummy,),str(destination/'extractor.onnx'),
                          input_names=['normalized_rgb'],output_names=['features'],opset_version=OPSET,
                          dynamo=True,external_data=False)
        graph = onnx.load(destination/'extractor.onnx')
        onnx.checker.check_model(graph,full_check=True)
        if next(o.version for o in graph.opset_import if o.domain=='') != OPSET:
            raise ValidationError('Exporter did not preserve requested opset')
        write_json(destination/'export.json',{'spec':EXPORT_SPEC,'feature_config':FEATURE_CONFIG,
            'model_metadata_sha256':sha256_file(model_path/'metadata.json'),
            'source_extractor_sha256':sha256_file(model_path/'extractor.pth'),
            'onnx_sha256':sha256_file(destination/'extractor.onnx'),
            'data_kind':model.metadata['data_kind'],'versions':{n:version(n) for n in ['torch','onnx','onnxscript','onnxruntime']},
            'parity_tolerances':PARITY,'weight_license':model.metadata['weights']['license_status']})
        runtime = OnnxExtractor(destination,model_path)
        test = np.random.default_rng(42).random((3,256,256),dtype=np.float32)
        a,b=model.extractor.embeddings(test),runtime.embeddings(test)
        passed=bool(np.allclose(a,b,atol=PARITY['embedding_atol'],rtol=PARITY['embedding_rtol']))
        write_json(destination/'export_check.json',{'synthetic_input_only':True,'embedding_allclose':passed,'max_absolute_error':float(np.max(np.abs(a-b)))})
        if not passed: raise ValidationError('ONNX export failed synthetic embedding parity')
    except Exception as exc:
        write_json(destination/'status.json',{'status':'failed','reason':str(exc)})
        raise ValidationError(f'ONNX export failed: {exc}') from exc
    write_json(destination/'status.json',{'status':'complete'})
    return {'output':str(destination),'scope':EXPORT_SPEC['scope'],'embedding_parity':passed}
