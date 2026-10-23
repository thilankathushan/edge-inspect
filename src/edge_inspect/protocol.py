"""Prespecified choices; changing these requires a new protocol version."""
import math
from .config import ValidationError

PROTOCOL = {
    "version": "bottle-v1",
    "calibration_fraction": 0.20,
    "target_fpr": 0.05,
    "image_size": [256, 256],
    "image_resize": "Pillow bilinear, direct resize, no crop",
    "mask_resize": "Pillow nearest, binary",
    "baseline_normalization": "RGB float32 CHW / 255",
    "feature_mean": [0.485, 0.456, 0.406],
    "feature_std": [0.229, 0.224, 0.225],
    "baseline": "model_fit mean RGB template; mean-channel squared residual; max score",
    "pretrained": "frozen ResNet18 IMAGENET1K_V1 layer2; all fit patches; L2 nearest neighbor; max score",
}


def normal_threshold(scores, alpha=0.05):
    """Use only held-out normal calibration scores from a fixed scoring rule."""
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not 0 < alpha < 1:
        raise ValidationError("target FPR must be between 0 and 1")
    scores = list(scores)
    if not scores or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
        raise ValidationError("calibration scores must be nonempty finite numbers")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return sorted(scores)[k - 1] if k <= len(scores) else math.inf
