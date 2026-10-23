"""End-to-end SYNTHETIC SOFTWARE verification, never an MVTec benchmark."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from edge_inspect.baseline import output_directory, train
from edge_inspect.config import load_config
from edge_inspect.manifest import write_manifest
from edge_inspect.patchcore import train_features
from edge_inspect.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='fixture-evaluation')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = output_directory(project / 'artifacts', args.name)
    root.mkdir(parents=True)
    (root / 'README.txt').write_text('SYNTHETIC SOFTWARE FIXTURES ONLY. No real benchmark results.\n')
    config_path = root / 'config.toml'
    config_path.write_text('data_root="data"\ncategory="bottle"\nseed=42\nartifact_root="outputs"\ndevice="cpu"\n')
    config = load_config(config_path)
    category = config.data_root / 'bottle'
    for part in ['train/good','test/good','test/fixture','ground_truth/fixture']:
        (category / part).mkdir(parents=True)
    for i in range(100):
        a = np.full((32,32,3),100+i%7,dtype=np.uint8)
        a[0,0] = [i,100,100]
        Image.fromarray(a).save(category / f'train/good/{i:03}.png')
    for i,level in enumerate([102,104,20]):
        Image.new('RGB',(32,32),(level,level,level)).save(category / f'test/good/{i:03}.png')
    for i,color in enumerate([(104,104,104),(160,100,100),(255,0,0)]):
        a = Image.new('RGB',(32,32),(103,103,103))
        a.paste(color,(12,12,20,20))
        a.save(category / f'test/fixture/{i:03}.png')
        mask=Image.new('L',(32,32),0)
        mask.paste(255,(12,12,20,20))
        mask.save(category / f'ground_truth/fixture/{i:03}_mask.png')
    manifest = Path(write_manifest(config)['manifest'])
    baseline = Path(train(config,manifest,'baseline','synthetic_fixture')['model'])
    features = Path(train_features(config,manifest,'features','synthetic_fixture',project / 'artifacts/weights/resnet18-f37072fd.pth')['model'])
    result = evaluate(config,manifest,baseline,features,project / 'configs/frozen_comparison.json','comparison',fixture=True)
    print('SYNTHETIC FIXTURE VERIFICATION ONLY')
    print(result)

if __name__ == '__main__': main()
