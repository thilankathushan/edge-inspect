"""Static evaluation plots and deterministic diagnostic examples, never model tuning."""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .config import ValidationError


def reporting_ready(artifact_root):
    os.environ.setdefault('MPLCONFIGDIR', str(artifact_root / '.matplotlib'))
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError as exc:
        raise ValidationError("Install plotting support: .venv/bin/python -m pip install -e '.[evaluation]'") from exc


def number(value):
    return 'undefined' if value is None else f'{value:.4f}'


def representatives(records):
    result = {}
    for outcome in ['TP', 'FN', 'FP', 'TN', 'uncalibrated']:
        group = sorted((r for r in records if r['outcome'] == outcome), key=lambda r: (r['score'], r['path']))
        if group:
            result[outcome] = group[(len(group) - 1) // 2]
    return result


def diagnostic_overlay(destination, row, name, data_root, scale, title):
    with Image.open(data_root / row['path']) as source:
        original = source.convert('RGB')
    mask = Image.new('L', original.size, 0)
    if row['mask_path']:
        with Image.open(data_root / row['mask_path']) as source:
            mask = source.convert('L').copy()
    size = (320, max(1, round(320 * original.height / original.width)))
    if size[1] > 320:
        size = (max(1, round(320 * original.width / original.height)), 320)
    preview = original.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    amap = np.load(destination / row['map_path'], allow_pickle=False)
    display = np.asarray(Image.fromarray(amap).resize(size, Image.Resampling.BILINEAR))
    alpha = (0.65 * np.sqrt(np.clip(display / max(scale, 1e-6), 0, 1)))[..., None]
    colored = ((1 - alpha) * np.asarray(preview) + alpha * np.array([255, 80, 0])).astype(np.uint8)
    canvas = Image.new('RGB', (990, size[1] + 120), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill='black')
    draw.text((10, 28), f"{name} | {row['outcome']} | score={row['score']:.6g} | pixel AUROC={number(row['pixel_auroc']['value'])}", fill='black')
    draw.text((10, 48), row['path'][:145], fill='black')
    draw.text((10, 68), f'Orange distance/residual overlay, not probability. Display scale 0..{scale:.6g}; no pixel threshold.', fill='black')
    for x, label, panel in [(10, 'Input', preview), (340, 'Ground-truth mask (white = defect)', mask.convert('RGB')), (670, 'Anomaly overlay', Image.fromarray(colored))]:
        draw.text((x, 92), label, fill='black')
        canvas.paste(panel, (x, 112))
    target = destination / name / f"{row['outcome']}_representative.png"
    canvas.save(target)
    return target.relative_to(destination).as_posix()


def create_report(destination, results, snapshot, data_root):
    import matplotlib.pyplot as plt
    mode = results['mode']
    title = 'SYNTHETIC FIXTURE VERIFICATION — NOT MVTEC RESULTS' if mode == 'synthetic_fixture_verification' else ('EXPLORATORY — TEST SET PREVIOUSLY EXPOSED' if mode == 'exploratory' else 'Frozen held-out MVTec bottle comparison')
    models = results['models']
    colors = {'baseline': '#276FBF', 'patchcore_style': '#D16B27'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name, value in models.items():
        summary = value['summary']
        roc = summary['roc']
        if roc['fpr']:
            axes[0].plot(roc['fpr'], roc['tpr'], label=f"{name}: {number(summary['image_auroc']['value'])}", color=colors[name])
    axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    axes[0].set(xlabel='False-positive rate', ylabel='Defect recall', title='Image ROC (threshold independent)', xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    keys = ['Image AUROC', 'Defect recall', 'FPR (lower is better)', 'Pixel macro AUROC']
    for i, (name, value) in enumerate(models.items()):
        s = value['summary']
        vals = [s['image_auroc']['value'], s['operating']['defect_recall']['value'], s['operating']['false_positive_rate']['value'], s['pixel_macro']['auroc']['value']]
        x = np.arange(4) + (i - 0.5) * 0.35
        axes[1].bar(x, [np.nan if v is None else v for v in vals], width=0.35, label=name, color=colors[name])
        for xx, v in zip(x, vals):
            if v is None:
                axes[1].text(xx, 0.04, 'undefined', rotation=90, fontsize=7, ha='center')
    axes[1].set_xticks(np.arange(4), keys, rotation=18, ha='right', fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(destination / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (name, value) in zip(axes, models.items()):
        for label, color, marker in [(False, '#2C8B57', 'o'), (True, '#BF3838', 'x')]:
            group = [r for r in value['records'] if r['is_anomaly'] == label]
            ax.scatter(range(len(group)), [r['score'] for r in group], label='defect' if label else 'normal', color=color, marker=marker, s=28)
        if value['threshold'] is not None:
            ax.axhline(value['threshold'], color='black', linestyle='--', label='saved calibration threshold')
        else:
            ax.text(0.02, 0.95, 'No finite threshold', transform=ax.transAxes, va='top')
        ax.set(title=name, ylabel='Image anomaly score', xlabel='Index within class (manifest order)')
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(destination / 'scores.png', dpi=160)
    plt.close(fig)
    lines = [f'# {title}', '', f"Mode: `{mode}`. Snapshot recorded before inference: `{snapshot['recorded_before_test_inference_utc']}`.",
             '', 'No fitting, recalibration, or hyperparameter selection occurred in this evaluation. Later improvements must be reported as exploratory. Scores from this report must not be used to retune and call the same test set untouched.',
             '', '![Comparison](comparison.png)', '', '![Scores](scores.png)', '',
             '| Model | Image AUROC | Recall | FPR | TP / FN / FP / TN | Defect / normal n | Pixel macro AUROC (eligible/excluded) |',
             '|---|---:|---:|---:|---|---|---|']
    for name, value in models.items():
        s = value['summary']; op = s['operating']; pix = s['pixel_macro']
        lines.append(f"| {name} | {number(s['image_auroc']['value'])} | {number(op['defect_recall']['value'])} | {number(op['false_positive_rate']['value'])} | {op['tp']} / {op['fn']} / {op['fp']} / {op['tn']} | {op['defect_count']} / {op['normal_count']} | {number(pix['auroc']['value'])} ({pix['eligible_images']}/{pix['excluded_images']}) |")
    b = models['baseline']['summary']['image_auroc']['value']
    p = models['patchcore_style']['summary']['image_auroc']['value']
    if mode == 'synthetic_fixture_verification':
        lines += ['', '**These figures only exercise evaluator software. They provide no evidence that either model is better on real MVTec defects.**']
    elif b is None or p is None:
        lines += ['', 'Image-AUROC comparison is undefined; no winner can be stated.']
    else:
        comparison = 'exceeds' if p > b else 'does not exceed'
        lines += ['', f'PatchCore-style image AUROC {comparison} the baseline (difference {p - b:+.6f}). Recall/FPR and localization are separate outcomes; no post-hoc tuning was performed.']
    lines += ['', '## Pixel aggregation and undefined values', '',
              'Pixel AUROC is exact per image on original-resolution masks after bilinear map resizing, with ties worth half credit. The reported value is the unweighted mean over images containing both foreground and background. Normal images and all-foreground masks have undefined per-image AUROC and are counted as excluded. This is not pooled/global pixel AUROC and must not be compared directly to pooled benchmark numbers.',
              '', 'All undefined values have null plus an explanation in results.json. Empty class denominators are not reported as zero. No finite calibration threshold means undefined operating decisions/confusion counts, while ranking metrics can still be defined.']
    for name, value in models.items():
        s = value['summary']
        for label, m in [('image AUROC', s['image_auroc']), ('recall', s['operating']['defect_recall']), ('FPR', s['operating']['false_positive_rate']), ('pixel macro AUROC', s['pixel_macro']['auroc'])]:
            if m['value'] is None:
                lines.append(f"- {name}, {label}: {m['reason']}.")
    lines += ['', '## Per-defect results', '', 'Each type uses the same normal test images as negatives. Small groups are descriptive, not statistically stable. Per-type AUROC requires at least two samples in each class.', '',
              '| Model / type | Defect n | TP / FN | Recall | Type-v-normal AUROC | Pixel macro AUROC |', '|---|---:|---|---:|---:|---:|']
    for name, value in models.items():
        for label, s in value['summary']['per_defect_type'].items():
            op = s['operating']
            lines.append(f"| {name} / {label} | {op['defect_count']} | {op['tp']} / {op['fn']} | {number(op['defect_recall']['value'])} | {number(s['image_auroc']['value'])} | {number(s['pixel_macro']['auroc']['value'])} |")
    lines += ['', '## Latency', '', f"Hardware: `{snapshot['hardware']}`.", '',
              'CPU, 256×256 input, batch size 1, one torch thread, five warmups on a model-fit normal. Each test image is timed once with perf_counter_ns. Model-only includes score/map generation; pipeline timing adds validation, decoding, resize and normalization. Model loading, hashes, masks, original-resolution resize, metrics, plotting and writes are excluded. OS cache state is uncontrolled and files have been read for integrity checking. These are warm-run observations, not cold-start benchmarks. Model order is baseline then PatchCore-style. Small n gives unstable percentiles; no memory benchmark is claimed.', '',
              '| Model | Timed n | Model-only p50 / p95 ms | Decode+preprocess+model p50 / p95 ms |', '|---|---:|---|---|']
    for name, value in models.items():
        m = value['summary']['latency']['model_only']; pipe = value['summary']['latency']['decode_preprocess_model']
        lines.append(f"| {name} | {m['count']} | {number(m['p50_ms'])} / {number(m['p95_ms'])} | {number(pipe['p50_ms'])} / {number(pipe['p95_ms'])} |")
    lines += ['', '## Error analysis and representative examples', '',
              'The selection rule was fixed before inference: median-score example per outcome bucket, breaking ties by path. All predictions remain available; these examples do not replace the full confusion counts. Ground-truth masks are shown for localization comparison. Observations below describe score/annotation behavior; proposed causes are hypotheses, not established mechanisms.']
    for name, value in models.items():
        op = value['summary']['operating']
        lines += ['', f'### {name}', '', f"Missed defects: {op['fn']}; false positives: {op['fp']}. Threshold: {value['threshold']}; calibration normals: {value['calibration_count']}."]
        selected = representatives(value['records'])
        missing = [x for x in ['TP', 'FN', 'FP', 'TN'] if x not in selected]
        if missing:
            lines.append('No representative exists for: ' + ', '.join(missing) + '.')
        scale = 1.0 if name == 'baseline' else max(value['threshold'] or max((r['score'] for r in snapshot['models'][name]['metadata']['calibration']['scores']), default=1), 1e-6)
        for outcome, row in selected.items():
            link = diagnostic_overlay(destination, row, name, data_root, scale, title)
            lines += ['', f"**{outcome}: {row['path']}** — score {row['score']:.6g}, saved threshold {value['threshold']}. "]
            if outcome == 'FN':
                lines.append('The annotated defect did not exceed the normal-calibrated image threshold. Small/low-contrast regions can be suppressed by resizing or feature receptive fields; the overlay shows whether the residual is weak or displaced. This explanation is a hypothesis, not a verified cause.')
            elif outcome == 'FP':
                lines.append('A normal-labeled image exceeded the threshold. Appearance variation, alignment, illumination or incomplete normal coverage are possible causes; an image score alone cannot distinguish them.')
            if row['is_anomaly']:
                lines.append(f"Localization evidence: peak inside annotation={row['peak_inside_defect']}; mean defect-region score={number(row['defect_region_mean'])}; background mean={number(row['background_region_mean'])}; within-image pixel AUROC={number(row['pixel_auroc']['value'])}.")
            lines += ['', f'![{name} {outcome}]({link})']
    lines += ['', '## Interpretation limits and next steps', '',
              'The baseline is sensitive to alignment and lighting; its maximum residual can be dominated by noise. The feature detector can miss normal modes through bounded sampling and has coarse receptive-field localization. Both maps are resized, not pixel-calibrated probabilities. Pixel macro AUROC weights eligible images equally and omits one-class masks. Small per-type samples and correlated images limit generalization. No exploratory improvements are included in this report.',
              '', 'Keep this snapshot, manifest, artifacts and results. Any later change informed by these outcomes must be labeled exploratory and needs a new untouched test set for confirmatory claims.']
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n')
