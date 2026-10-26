"""Prepare only generated fixtures for the free public demo; no downloads."""
from pathlib import Path
import subprocess
import sys
from edge_inspect.baseline import TemplateModel

root=Path(__file__).resolve().parents[1]
model=root/'artifacts/public-demo/outputs/models/fixture-model'
if not model.exists():
    subprocess.run([sys.executable,str(root/'scripts/verify_fixture_baseline.py'),'--name','public-demo'],check=True)
loaded=TemplateModel.load(model)
if loaded.metadata['data_kind']!='synthetic_fixture':
    raise RuntimeError('Public demo requires a generated-fixture model')
print('Public fixture baseline validated. No real dataset or pretrained weights included.')
