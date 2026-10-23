"""Exact binary ranking metrics, explicit undefined values, and descriptive summaries."""
import math
import numpy as np
from .config import ValidationError


def metric(value=None, reason=None):
    return {'value': value, 'reason': reason}


def binary_inputs(labels, scores):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValidationError('Metrics require equal-length 1D binary labels and finite scores')
    return y.astype(np.int64), s


def auroc(labels, scores):
    """P(score_positive > score_negative) + 0.5 P(tie), exact without binning."""
    y, s = binary_inputs(labels, scores)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        return metric(reason='AUROC requires at least one positive and one negative')
    order = np.argsort(s, kind='stable')
    s, y = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, len(y)])
    pos = np.add.reduceat(y, starts)
    neg = counts - pos
    neg_before = np.cumsum(neg, dtype=np.float64) - neg
    wins = np.sum(pos * (neg_before + 0.5 * neg), dtype=np.float64)
    return metric(float(wins / (positives * negatives)))


def roc_curve(labels, scores):
    y, s = binary_inputs(labels, scores)
    p, n = int(y.sum()), int(len(y) - y.sum())
    if not p or not n:
        return {'fpr': [], 'tpr': [], 'reason': 'ROC requires both classes'}
    order = np.argsort(-s, kind='stable')
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    return {'fpr': np.r_[0, fp / n].tolist(), 'tpr': np.r_[0, tp / p].tolist(), 'reason': None}


def operating_metrics(labels, scores, threshold):
    y, s = binary_inputs(labels, scores)
    p, n = int(y.sum()), int(len(y) - y.sum())
    if threshold is None or not math.isfinite(threshold):
        reason = 'No finite calibration-selected threshold; decisions are uncalibrated'
        return {'sample_count': len(y), 'defect_count': p, 'normal_count': n,
                'tp': None, 'fn': None, 'fp': None, 'tn': None,
                'defect_recall': metric(reason=reason), 'false_positive_rate': metric(reason=reason)}
    predictions = s > threshold  # Same strict comparison as model calibration.
    tp, fp = int(np.sum(predictions & (y == 1))), int(np.sum(predictions & (y == 0)))
    return {'sample_count': len(y), 'defect_count': p, 'normal_count': n,
            'tp': tp, 'fn': p - tp, 'fp': fp, 'tn': n - fp,
            'defect_recall': metric(tp / p) if p else metric(reason='No defect samples'),
            'false_positive_rate': metric(fp / n) if n else metric(reason='No normal samples')}


def pixel_macro(records):
    valid = [r['pixel_auroc']['value'] for r in records if r['pixel_auroc']['value'] is not None]
    return {'auroc': metric(float(np.mean(valid))) if valid else metric(reason='No image contains both defect and background pixels'),
            'aggregation': 'Unweighted mean of exact within-image AUROCs at original resolution; not pooled pixel AUROC',
            'eligible_images': len(valid), 'excluded_images': len(records) - len(valid),
            'positive_pixels': sum(r['positive_pixels'] for r in records),
            'negative_pixels': sum(r['negative_pixels'] for r in records)}


def latency_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {'count': 0, 'p50_ms': None, 'p95_ms': None, 'reason': 'No measurements'}
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValidationError('Invalid latency samples')
    return {'count': len(array), 'p50_ms': float(np.percentile(array, 50)),
            'p95_ms': float(np.percentile(array, 95)), 'mean_ms': float(array.mean()),
            'min_ms': float(array.min()), 'max_ms': float(array.max()), 'reason': None}


def summarize(records, threshold):
    labels = [int(r['is_anomaly']) for r in records]
    scores = [r['score'] for r in records]
    normals = [r for r in records if not r['is_anomaly']]
    per_type = {}
    for label in sorted({r['label'] for r in records if r['is_anomaly']}):
        defects = [r for r in records if r['label'] == label]
        group = normals + defects
        yy, ss = [int(r['is_anomaly']) for r in group], [r['score'] for r in group]
        per_type[label] = {
            'operating': operating_metrics(yy, ss, threshold),
            'image_auroc': auroc(yy, ss) if len(normals) >= 2 and len(defects) >= 2 else metric(reason='Per-type AUROC requires at least two normals and two defects'),
            'pixel_macro': pixel_macro(defects),
            'caution': 'Small descriptive sample; fewer than five defects' if len(defects) < 5 else 'Descriptive results; shared normal denominator across defect types',
        }
    return {'image_auroc': auroc(labels, scores), 'operating': operating_metrics(labels, scores, threshold),
            'pixel_macro': pixel_macro(records), 'per_defect_type': per_type,
            'roc': roc_curve(labels, scores),
            'latency': {'model_only': latency_summary([r['model_ms'] for r in records]),
                        'decode_preprocess_model': latency_summary([r['pipeline_ms'] for r in records])}}
