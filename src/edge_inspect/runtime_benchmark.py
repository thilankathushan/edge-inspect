"""Isolated CPU runtime measurements; no fitting, threshold adjustment or implicit download."""
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter_ns
from importlib.metadata import version
import numpy as np
from PIL import Image
from .baseline import output_directory, read_json, write_json, training_records
from .config import ValidationError
from .data import sha256_file, validate_image
from .evaluation import hardware_info, utc_now
from .manifest import select_records, verified_path
from .metrics import auroc, summarize, latency_summary
from .onnx_backend import PARITY, EXPORT_SPEC
from .preprocessing import preprocess_image

BENCHMARK_PROTOCOL = {
    'version':'runtime-comparison-v1', 'input':[1,3,256,256], 'precision':'FP32',
    'provider':'CPUExecutionProvider', 'torch_intra_threads':1, 'torch_inter_threads':1,
    'ort_intra_threads':1,'ort_inter_threads':1,'ort_execution':'sequential',
    'ort_optimization':'ORT_ENABLE_ALL','ort_spinning':False,
    'timing':'perf_counter_ns; warm disk caches; exclude load/hash/masks/metrics/output',
    'components': {'preprocessing':'validate/decode, RGB bilinear resize, /255 CHW',
                   'embedding':'ImageNet normalization + backbone + NumPy patch layout',
                   'scoring':'unchanged torch exact chunked 1-NN + map interpolation',
                   'end_to_end':'preprocessing through score/map; no serialization or original-resolution rendering'},
    'memory':'Fresh process per runtime. psutil current RSS before model load, after load, after warmup and after measurements. resource ru_maxrss process-lifetime high-water mark (macOS bytes, Linux KiB converted). Includes Python/framework imports, source validation, load, caches, parity array saving and pixel-metric work; not incremental model allocation, GPU memory or whole-system RAM. Stage RSS can miss transient peaks; high-water includes startup. Shared pages counted; allocator retention and OS effects apply.',
    'order':'original then ONNX in separate sequential processes; no concurrency, no cold-start claim',
    'parity':PARITY,'threshold':'unchanged original normal-calibration threshold; strict >; no retuning',
    'quality':'same exact image and original-resolution macro per-image pixel metrics as frozen evaluator; no anomalies in calibration => image AUROC undefined',
}


def decision(score, threshold):
    return bool(score > threshold) if math.isfinite(threshold) else None


def parity_record(a, b, aa, bb, amap, bmap, threshold):
    def close(x,y,kind):
        return bool(np.allclose(x,y,atol=PARITY[kind+'_atol'],rtol=PARITY[kind+'_rtol']))
    return {'embedding_allclose':close(aa,bb,'embedding'), 'embedding_max_abs_error':float(np.max(np.abs(aa-bb))),
            'score_allclose':close(a,b,'score'),'score_abs_error':abs(a-b),
            'map_allclose':close(amap,bmap,'map'),'map_max_abs_error':float(np.max(np.abs(amap-bmap))),
            'original_score':a,'onnx_score':b,'original_decision':decision(a,threshold),'onnx_decision':decision(b,threshold),
            'decision_changed':decision(a,threshold)!=decision(b,threshold)}


def worker(job_path, backend):
    import psutil
    import torch
    from .patchcore import PatchModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    job=read_json(job_path)
    destination=Path(job['output'])/backend
    destination.mkdir()
    process=psutil.Process()
    memory={'before_model_load_rss_bytes':process.memory_info().rss}
    model=PatchModel.load(Path(job['model']),onnx_path=Path(job['onnx']) if backend=='onnx' else None)
    memory['after_load_rss_bytes']=process.memory_info().rss
    data_root=Path(job['data_root'])
    warm=preprocess_image(verified_path(data_root,job['warmup_row']['path'],job['warmup_row']['sha256']))
    for _ in range(job['warmup']): model.score(warm)
    memory['after_warmup_rss_bytes']=process.memory_info().rss
    rows=job['rows']
    # Verify immutable input bytes outside the timer before reading any sample.
    paths=[verified_path(data_root,r['path'],r['sha256']) for r in rows]
    samples={key:[] for key in ['preprocessing','embedding','scoring','end_to_end']}
    records=[]
    raw=[]
    for repetition in range(job['repeats']):
        for i,(row,path) in enumerate(zip(rows,paths)):
            begin=perf_counter_ns()
            image=preprocess_image(path)
            prep_end=perf_counter_ns()
            patches=model.extractor.embeddings(image)
            embed_end=perf_counter_ns()
            score,small_map=model.score_patches(patches)
            end=perf_counter_ns()
            values=[(prep_end-begin)/1e6,(embed_end-prep_end)/1e6,(end-embed_end)/1e6,(end-begin)/1e6]
            for key,value in zip(samples,values): samples[key].append(value)
            raw.append({'path':row['path'],'repetition':repetition,**dict(zip(samples,values)), 'score':score})
            if repetition: continue
            np.save(destination/f'{i:04}-embeddings.npy',patches,allow_pickle=False)
            np.save(destination/f'{i:04}-map.npy',small_map,allow_pickle=False)
            mask=np.zeros((row['height'],row['width']),dtype=bool)
            if row['mask_path']:
                mask_path=verified_path(data_root,row['mask_path'],row['mask_sha256'])
                validate_image(mask_path,mask=True)
                with Image.open(mask_path) as source: mask=np.asarray(source.convert('L'))>0
                if mask.shape!=(row['height'],row['width']): raise ValidationError('Mask dimensions changed')
            full=np.asarray(Image.fromarray(small_map).resize((row['width'],row['height']),Image.Resampling.BILINEAR))
            records.append({'path':row['path'],'label':row['label'],'is_anomaly':row['is_anomaly'],
                            'score':score,'decision':decision(score,model.threshold),'pixel_auroc':auroc(mask.ravel(),full.ravel()),
                            'positive_pixels':int(mask.sum()),'negative_pixels':int((~mask).sum()),
                            'model_ms':values[1]+values[2],'pipeline_ms':values[3]})
    memory['after_measurements_rss_bytes']=process.memory_info().rss
    memory['process_lifetime_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)
    write_json(destination/'result.json',{'backend':backend,'records':records,'quality':summarize(records,model.threshold),
        'latency':{key:latency_summary(values) for key,values in samples.items()},'raw_measurements':raw,'memory':memory,
        'provider':model.extractor.session.get_providers() if backend=='onnx' else ['PyTorch CPU'],
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()})


def benchmark(config, manifest_path, model_path, onnx_path, run_name, *, warmup=5, repeats=5, evaluation=None):
    if warmup<1 or repeats<2: raise ValidationError('Benchmark requires warmup >= 1 and repeats >= 2')
    from .onnx_backend import OnnxExtractor
    manifest=read_json(manifest_path)
    fit,cal,_=training_records(config,manifest)
    meta=read_json(model_path/'metadata.json')
    if meta.get('manifest_sha256')!=sha256_file(manifest_path) or meta.get('category')!=config.category or meta.get('seed')!=config.seed:
        raise ValidationError('Model and benchmark manifest/configuration disagree')
    for saved,expected in [(meta['fit_records'],fit),(meta['calibration']['scores'],cal)]:
        if sorted((r['path'],r['sha256']) for r in saved)!=sorted((r['path'],r['sha256']) for r in expected):
            raise ValidationError('Model split provenance disagrees with manifest')
    # Load validates source binding and export hash, without opening evaluation images.
    runtime=OnnxExtractor(onnx_path,model_path)
    del runtime
    mode='normal_calibration_runtime_check'
    rows=cal
    if evaluation is not None:
        status=read_json(evaluation/'status.json')
        snapshot=read_json(evaluation/'prerun_snapshot.json')
        previous=read_json(evaluation/'results.json')
        if status.get('status')!='complete' or snapshot['manifest_sha256']!=sha256_file(manifest_path) or snapshot['models']['patchcore_style']['files']['metadata.json']!=sha256_file(model_path/'metadata.json') or previous['snapshot_sha256']!=sha256_file(evaluation/'prerun_snapshot.json'):
            raise ValidationError('Test runtime check requires a completed frozen evaluation of this exact model and manifest')
        rows=sorted(select_records(manifest,'final_test',allow_final_test=True),key=lambda r:r['path'])
        mode='exploratory_runtime_comparison_of_previously_evaluated_test'
    if meta['data_kind']=='synthetic_fixture': mode='synthetic_fixture_verification_'+mode
    destination=output_directory(config.artifact_root/'benchmarks',run_name)
    destination.mkdir(parents=True)
    hashes={name:sha256_file(model_path/name) for name in ['metadata.json','extractor.pth','memory_bank.npy']}
    snapshot={'recorded_before_inference_utc':utc_now(),'mode':mode,'data_kind':meta['data_kind'],
              'model_files':hashes,'model_metadata':meta,'manifest_sha256':sha256_file(manifest_path),
              'export':read_json(onnx_path/'export.json'),'protocol':BENCHMARK_PROTOCOL,'warmup':warmup,'repeats':repeats,
              'hardware':hardware_info(),'versions':{n:version(n) for n in ['numpy','Pillow','torch','torchvision','onnx','onnxscript','onnxruntime','psutil']},
              'evaluation_source':str(evaluation.resolve()) if evaluation else None,
              'source_sha256':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(destination/'prerun_snapshot.json',snapshot)
    job={'output':str(destination.resolve()),'model':str(model_path.resolve()),'onnx':str(onnx_path.resolve()),
         'data_root':str(config.data_root),'rows':rows,'warmup_row':fit[0],'warmup':warmup,'repeats':repeats}
    write_json(destination/'job.json',job)
    write_json(destination/'status.json',{'status':'running'})
    try:
        for backend in ['original','onnx']:
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
            completed=subprocess.run([sys.executable,'-m','edge_inspect.runtime_benchmark',str(destination/'job.json'),backend],env=env,capture_output=True,text=True)
            (destination/f'{backend}-worker.log').write_text(completed.stdout+completed.stderr)
            if completed.returncode: raise ValidationError(f'{backend} worker failed; see {destination}/{backend}-worker.log')
        results={name:read_json(destination/name/'result.json') for name in ['original','onnx']}
        parity=[]
        threshold=meta['calibration']['threshold']
        threshold=threshold if threshold is not None else math.inf
        for i,(a,b) in enumerate(zip(results['original']['records'],results['onnx']['records'])):
            arrays=[np.load(destination/backend/f'{i:04}-{kind}.npy',allow_pickle=False) for kind in ['embeddings','map'] for backend in ['original','onnx']]
            parity.append({'path':a['path'],**parity_record(a['score'],b['score'],*arrays,threshold)})
        if hashes!={name:sha256_file(model_path/name) for name in hashes} or sha256_file(manifest_path)!=snapshot['manifest_sha256'] or sha256_file(onnx_path/'extractor.onnx')!=snapshot['export']['onnx_sha256']:
            raise ValidationError('Frozen inputs changed during benchmark')
        result={'mode':mode,'runtimes':results,'parity':parity,'decision_changes':sum(r['decision_changed'] for r in parity),
                'numerical_parity_passed':all(r['embedding_allclose'] and r['score_allclose'] and r['map_allclose'] for r in parity),
                'snapshot_sha256':sha256_file(destination/'prerun_snapshot.json')}
        write_json(destination/'results.json',result)
        report(destination,result,snapshot)
        write_json(destination/'status.json',{'status':'complete'})
        return {'output':str(destination),'mode':mode,'numerical_parity_passed':result['numerical_parity_passed'],'decision_changes':result['decision_changes']}
    except Exception as exc:
        write_json(destination/'status.json',{'status':'failed','reason':str(exc)})
        raise


def report(destination,result,snapshot):
    lines=['# ONNX extractor runtime comparison','',f"Mode: `{result['mode']}`.",'',
           '**SYNTHETIC FIXTURE measurements, not real MVTec quality or edge-device benchmarks.**' if snapshot['data_kind']=='synthetic_fixture' else 'Observed on the recorded local hardware only.',
           '',f"Hardware: {snapshot['hardware']}",f"Versions: {snapshot['versions']}",'',
           'Only ResNet18 through layer2 is ONNX. Preprocessing/normalization, patch layout, torch nearest-neighbor scoring, map resize and the unchanged threshold remain external. No quantization.',
           '',f"Warmup: {snapshot['warmup']} full calls on a fit normal; repeats: {snapshot['repeats']} per input; batch 1, 256x256 FP32, one intra/inter-op thread, CPUExecutionProvider, sequential ORT, full graph optimization, spinning disabled.",
           '', '| Component | Original p50 / p95 ms | ONNX path p50 / p95 ms |','|---|---:|---:|']
    for key in ['preprocessing','embedding','scoring','end_to_end']:
        a,b=[result['runtimes'][n]['latency'][key] for n in ['original','onnx']]
        lines.append(f"| {key} | {a['p50_ms']:.3f} / {a['p95_ms']:.3f} | {b['p50_ms']:.3f} / {b['p95_ms']:.3f} |")
    a,b=[result['runtimes'][n]['latency']['end_to_end']['p50_ms'] for n in ['original','onnx']]
    lines+=['',f'Observed median end-to-end ratio original/ONNX: {a/b:.3f}x; ONNX path was '+('faster' if b<a else 'slower or equal')+' in this run. No confidence interval or universal speedup claim.',
            '', 'Timings exclude load, file hashing, masks, metrics and output writing. Embedding includes normalization/layout; scoring includes map resize. End-to-end means validated file decode through score/map, without serialization/overlay rendering. Caches are warm, backend order fixed, and repeated inputs are correlated.',
            '', '## Numerical parity and quality','',f"Numerical parity passed: {result['numerical_parity_passed']}. Fixed atol=rtol=1e-4 for embeddings, score and maps. Decision changes: **{result['decision_changes']} / {len(result['parity'])}** at the unchanged calibration threshold.",
            f"Maximum embedding absolute error: {max(r['embedding_max_abs_error'] for r in result['parity']):.8g}; score absolute error: {max(r['score_abs_error'] for r in result['parity']):.8g}.",
            '', '| Runtime | Image AUROC | Recall | FPR | TP/FN/FP/TN | Pixel macro AUROC |','|---|---:|---:|---:|---|---:|']
    for name,data in result['runtimes'].items():
        s=data['quality'];o=s['operating']
        lines.append(f"| {name} | {s['image_auroc']['value']} | {o['defect_recall']['value']} | {o['false_positive_rate']['value']} | {o['tp']}/{o['fn']}/{o['fp']}/{o['tn']} | {s['pixel_macro']['auroc']['value']} |")
    for r in result['parity']:
        if r['decision_changed']: lines.append(f"Changed decision: `{r['path']}` original={r['original_score']:.10g} / {r['original_decision']}, ONNX={r['onnx_score']:.10g} / {r['onnx_decision']}.")
    lines+=['','None means undefined; JSON contains reasons and counts. Calibration-only inputs cannot establish defect recall or AUROC. Pixel aggregation matches the frozen original-resolution macro protocol. Test-mode inputs were already evaluated: this is an exploratory runtime comparison, not a fresh holdout. No threshold is adjusted to hide changed decisions.',
            '', '## Process memory','',BENCHMARK_PROTOCOL['memory'],'', '| Runtime | Before load MiB | After load MiB | After warmup MiB | After run MiB | Lifetime peak MiB |','|---|---:|---:|---:|---:|---:|']
    for name,data in result['runtimes'].items():
        values=[data['memory'][k]/(1024**2) for k in ['before_model_load_rss_bytes','after_load_rss_bytes','after_warmup_rss_bytes','after_measurements_rss_bytes','process_lifetime_peak_rss_bytes']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.1f}' for v in values)+' |')
    lines+=['','Both paths still import PyTorch for scoring. ONNX is not a standalone lightweight full detector. No MPS, Raspberry Pi, or other edge hardware was benchmarked. Quantization is deferred: this FP32 comparison must be understood before a separately calibrated/evaluated tradeoff.']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': worker(Path(sys.argv[1]),sys.argv[2])
