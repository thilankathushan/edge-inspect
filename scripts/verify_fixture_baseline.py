"""Generate a reproducible SOFTWARE FIXTURE run, never a benchmark."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from edge_inspect.config import load_config
from edge_inspect.manifest import write_manifest
from edge_inspect.baseline import train, output_directory
from edge_inspect.prediction import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='fixture-baseline')
    args = parser.parse_args()
    root = output_directory(Path(__file__).resolve().parents[1] / 'artifacts', args.name)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'README.txt').write_text('SYNTHETIC SOFTWARE FIXTURES ONLY. Not real MVTec data or a benchmark.\n')
    config_path = root / 'config.toml'
    config_path.write_text('data_root = "data"\ncategory = "bottle"\nseed = 42\nartifact_root = "outputs"\ndevice = "cpu"\n')
    config = load_config(config_path)
    category = config.data_root / 'bottle'
    for split in ['train/good', 'test/good', 'test/synthetic', 'ground_truth/synthetic']:
        (category / split).mkdir(parents=True)
    for i in range(100):
        pixels = np.full((32, 32, 3), 100 + i % 7, dtype=np.uint8)
        pixels[0, 0] = [i, 100, 100]  # Unique source bytes, deterministic.
        Image.fromarray(pixels).save(category / f'train/good/{i:03}.png')
    Image.new('RGB', (32, 32), (20, 20, 20)).save(category / 'test/good/000.png')
    Image.new('RGB', (32, 32), (250, 0, 0)).save(category / 'test/synthetic/000.png')
    Image.new('L', (32, 32), 255).save(category / 'ground_truth/synthetic/000_mask.png')
    manifest = write_manifest(config)
    model = train(config, Path(manifest['manifest']), 'fixture-model', 'synthetic_fixture')
    image = Image.new('RGB', (48, 32), (103, 103, 103))
    image.paste((255, 0, 0), (20, 12, 28, 20))
    user_image = root / 'synthetic-user-input.png'
    image.save(user_image)
    result = predict(config, Path(model['model']), user_image, 'fixture-prediction')
    print('SYNTHETIC SOFTWARE VERIFICATION ONLY')
    print(f"Model: {model['model']}")
    print(f"Prediction artifacts: {result['output']}")
    print('No official test performance evaluated.')


if __name__ == '__main__':
    main()
