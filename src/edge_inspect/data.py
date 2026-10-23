"""Strict, read-only MVTec inspection. Test access here is integrity-only."""
from collections import Counter
import hashlib
from pathlib import Path
import warnings
from PIL import Image
from .config import Config, ValidationError

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_image(path: Path, *, mask=False) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width < 2 or image.height < 2 or max(image.size) > 8192 or image.width * image.height > 32_000_000:
                    raise ValidationError(f'Image dimensions must be 2..8192 per side and at most 32 million pixels: {path}')
                if image.format not in {'PNG', 'JPEG', 'BMP', 'TIFF'} or path.suffix.lower() not in EXTENSIONS:
                    raise ValidationError(f'Unsupported image format or extension: {path}')
                image.load()
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValidationError(f'Multiframe images are unsupported: {path}')
                if image.getexif().get(274, 1) != 1:
                    raise ValidationError(f'Nontrivial EXIF orientation is unsupported: {path}')
                allowed = {'1', 'L'} if mask else {'L', 'RGB'}
                if image.mode not in allowed:
                    raise ValidationError(f'Unsupported {"mask" if mask else "image"} mode {image.mode}: {path}')
                if mask:
                    histogram = image.convert('L').histogram()
                    if any(histogram[1:255]) or not histogram[255]:
                        raise ValidationError(f'Mask must be binary 0/255 with nonempty foreground: {path}')
                return {'width': image.width, 'height': image.height, 'mode': image.mode}
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f'Cannot decode image {path}: {exc}') from exc


def scan_data(config: Config) -> tuple[list[dict], dict]:
    category = config.data_root / config.category
    if not category.is_dir():
        raise ValidationError(f'Category directory not found: {category}. Set data_root/category in the config; no data is downloaded automatically.')
    if category.is_symlink():
        raise ValidationError(f'Category symlinks are not supported: {category}')
    counts = Counter()
    ignored = 0
    images, masks = {}, {}
    for path in sorted(category.rglob('*')):
        if path.is_symlink():
            raise ValidationError(f'Dataset symlinks are not supported: {path}')
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            ignored += 1
            continue
        parts = path.relative_to(category).parts
        if len(parts) != 3 or parts[0] not in {'train', 'test', 'ground_truth'}:
            raise ValidationError(f'Unexpected image layout: {path}; expected split/label/image')
        split, label, _ = parts
        if split == 'train' and label != 'good':
            raise ValidationError(f'Training must contain normal images only: {path}')
        if split == 'ground_truth' and (label == 'good' or not path.name.endswith('_mask.png')):
            raise ValidationError(f'Unexpected mask path: {path}')
        info = validate_image(path, mask=split == 'ground_truth')
        info.update(path=path.relative_to(config.data_root).as_posix(), sha256=sha256_file(path))
        counts[f'{split}/{label}'] += 1
        if split == 'ground_truth':
            masks[(label, path.stem.removesuffix('_mask'))] = info
        else:
            key = (split, label, path.stem)
            if key in images:
                raise ValidationError(f'Ambiguous image stem: {path}')
            images[key] = info
    if not counts['train/good']:
        raise ValidationError(f'No normal training images found in {category / "train/good"}')
    records, used_masks, seen_hashes = [], set(), {}
    for (split, label, stem), info in sorted(images.items()):
        anomaly = label != 'good'
        mask = None
        if split == 'test' and anomaly:
            key = (label, stem)
            if key not in masks:
                raise ValidationError(f'Missing mask for {info["path"]}')
            mask = masks[key]
            used_masks.add(key)
            if (mask['width'], mask['height']) != (info['width'], info['height']):
                raise ValidationError(f'Mask dimensions do not match image: {info["path"]}')
        if info['sha256'] in seen_hashes:
            raise ValidationError(f'Duplicate image bytes: {info["path"]} and {seen_hashes[info["sha256"]]}')
        seen_hashes[info['sha256']] = info['path']
        records.append({**info, 'category': config.category, 'official_split': split,
                        'label': label, 'is_anomaly': anomaly,
                        'mask_path': mask['path'] if mask else None,
                        'mask_sha256': mask['sha256'] if mask else None})
    if masks.keys() - used_masks:
        raise ValidationError(f'Orphan masks: {sorted(masks.keys() - used_masks)}')
    return records, {'category': config.category, 'root': str(category),
                     'counts': dict(sorted(counts.items())), 'ignored_files': ignored,
                     'scope': 'integrity only; no model fitting or evaluation'}


def inspect_data(config: Config) -> dict:
    return scan_data(config)[1]
