"""Start with .venv/bin/streamlit run app.py"""
from pathlib import Path
import streamlit as st
from edge_inspect.demo import discover,load_model,inspect_upload,sample
from edge_inspect.config import ValidationError
st.set_page_config(page_title='Edge Inspect',page_icon='◈',layout='wide')
st.caption('EDGE INSPECT  /  LOCAL VISUAL INSPECTION')
st.title('See where an image differs.')
st.markdown('Explore saved anomaly detectors with a score, spatial evidence, and a calibrated decision.')
models=discover(Path(__file__).parent/'artifacts')
with st.sidebar:
    st.header('Inspection setup')
    st.caption('CPU inference · local session · bottle category')
    if not models:
        st.warning('No model artifacts found. Train a baseline or feature model first; see README setup.')
        st.stop()
    selected=st.selectbox('Saved model',range(len(models)),format_func=lambda i: ('Baseline' if models[i][1]['method'].startswith('rgb') else 'PatchCore-style')+' / '+str(models[i][0].relative_to(Path(__file__).parent/'artifacts')))
    path,meta=models[selected]
    st.write('Training provenance')
    st.code('MVTec AD via third-party mirror' if meta.get('dataset_acquisition') else meta['data_kind'],language=None)
    if meta.get('dataset_acquisition'):
        st.caption('Mirror transfer verified; publisher authenticity not independently verified.')
    st.caption('Models are loaded from local artifacts. No predictions are simulated.')
if meta['data_kind']=='synthetic_fixture':
    st.warning('Fixture-trained model — software demonstration only. Not validated on real bottle defects.')
else:
    st.info('Bottle inspection only. Uncontrolled lighting, pose, and other objects fall outside the benchmark conditions.')
left,right=st.columns([2,1])
with left:
    uploaded=st.file_uploader('Upload an inspection image',type=['png','jpg','jpeg'],max_upload_size=10)
with right:
    use_sample=st.checkbox('Use a synthetic geometry sample')
    st.caption('Original program-generated shapes, not a bottle or benchmark image. No third-party image content; sample dedicated to CC0. No MVTec images bundled.')
content=sample() if use_sample else uploaded.getvalue() if uploaded else None
name='sample.png' if use_sample else uploaded.name if uploaded else ''
if content is None:
    st.info('Choose an image or the synthetic sample, then run inspection.')
if st.button('Run inspection',type='primary',disabled=content is None):
    try:
        with st.spinner('Loading saved model and inspecting image…'):
            model=load_model(path)
            result=inspect_upload(model,content,name)
        a,b,c,d=st.columns(4)
        a.metric('Anomaly score',f"{result['score']:.6g}")
        b.metric('Calibration threshold','Unavailable' if result['threshold'] is None else f"{result['threshold']:.6g}")
        c.metric('Decision',{'Above threshold':'Flagged','Within threshold':'Within','Uncalibrated':'Uncalibrated'}[result['decision']])
        d.metric('Score + map time',f"{result['model_ms']:.1f} ms")
        columns=st.columns(3)
        for col,key,label in zip(columns,['original','heat','overlay'],['Original image','Anomaly heatmap','Overlay']):
            with col:
                st.subheader(label);st.image(result[key],width='stretch')
        st.caption(f"Validation/decode/resize + score/map: {result['pipeline_ms']:.1f} ms. Single call, no warmup. Model load, upload staging, preliminary validation and visualization excluded. Not a benchmark.")
        st.info(f"Brighter orange means greater distance from the model's normal reference, not defect probability. Display uses a fixed calibration-derived scale 0–{result['scale']:.5g}, square-root contrast, clipped at the upper bound. The decision uses the raw image score, strictly greater than the saved normal-calibration threshold (target FPR 5%); it does not identify which pixels are defects.")
    except (ValidationError,OSError,RuntimeError) as exc:
        st.error(f'Inspection could not complete: {exc}')
