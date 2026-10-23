"""Local demo adapter. All scores come from saved models."""
import hashlib
import io
import math
from pathlib import Path
import tempfile
from time import perf_counter_ns
import numpy as np
from PIL import Image
from .baseline import TemplateModel, read_json
from .config import ValidationError
from .data import validate_image
from .preprocessing import preprocess_image

LIMIT=10*1024*1024

def discover(root):
    found=[]
    for path in sorted(root.rglob('metadata.json')):
        if path.parent.parent.name!='models': continue
        try:
            meta=read_json(path)
            if meta.get('method') in ['rgb-mean-template-v1','patchcore-style-resnet18-v1']:
                found.append((path.parent,meta))
        except ValidationError: pass
    return found

def load_model(path):
    if read_json(path/'metadata.json')['method']=='rgb-mean-template-v1':
        return TemplateModel.load(path)
    from .patchcore import PatchModel
    return PatchModel.load(path)

def sample():
    """Original geometric software fixture; no third-party image content."""
    image=Image.new('RGB',(320,320),(103,103,103))
    image.paste((230,70,40),(130,120,190,180))
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()

def inspect_upload(model, content, name):
    suffix=Path(name).suffix.lower()
    if suffix not in {'.png','.jpg','.jpeg'}: raise ValidationError('Choose a PNG or JPEG image.')
    if not content or len(content)>LIMIT: raise ValidationError('Image must contain data and be at most 10 MiB.')
    if hashlib.sha256(content).hexdigest() in model.metadata.get('reserved_test_sha256',[]):
        raise ValidationError('This image belongs to the reserved evaluation set. Use the evaluator instead.')
    with tempfile.TemporaryDirectory(prefix='edge-inspect-') as directory:
        path=Path(directory)/('upload'+suffix);path.write_bytes(content)
        validate_image(path)
        with Image.open(path) as source:
            if source.format not in {'PNG','JPEG'}: raise ValidationError('Decoded format must be PNG or JPEG.')
            original=source.convert('RGB');original.thumbnail((768,768))
        start=perf_counter_ns();tensor=preprocess_image(path);middle=perf_counter_ns()
        score,anomaly=model.score(tensor);end=perf_counter_ns()
    finite=math.isfinite(model.threshold)
    scale=max(model.threshold,1e-6) if finite else max((r['score'] for r in model.metadata['calibration']['scores']),default=1)
    scale=max(scale,1e-6)
    values=np.asarray(Image.fromarray(anomaly).resize(original.size,Image.Resampling.BILINEAR))
    intensity=np.sqrt(np.clip(values/scale,0,1))
    heat=np.stack([255*intensity,100*intensity,40*intensity],axis=-1).astype('uint8')
    alpha=(.65*intensity)[...,None]
    overlay=((1-alpha)*np.asarray(original)+alpha*np.array([255,100,40])).astype('uint8')
    return dict(score=score,threshold=model.threshold if finite else None,
                decision=('Above threshold' if score>model.threshold else 'Within threshold') if finite else 'Uncalibrated',
                original=original,heat=heat,overlay=overlay,scale=scale,
                model_ms=(end-middle)/1e6,pipeline_ms=(end-start)/1e6)
