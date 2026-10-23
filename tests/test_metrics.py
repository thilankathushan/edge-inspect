"""Known-answer software tests; no benchmark data."""
import numpy as np
import pytest
from edge_inspect.metrics import auroc, roc_curve, operating_metrics, pixel_macro, latency_summary
from edge_inspect.config import ValidationError

@pytest.mark.parametrize('scores,expected', [([0,1,2,3],1),([3,2,1,0],0),([1,1,1,1],.5),([.1,.4,.35,.8],.75)])
def test_auc_known(scores, expected):
    assert auroc([0,0,1,1], scores)['value'] == expected
    curve = roc_curve([0,0,1,1], scores)
    assert np.trapezoid(curve['tpr'], curve['fpr']) == expected

@pytest.mark.parametrize('labels,scores', [([],[]),([1,1],[0,1]),([0],[1])])
def test_undefined_auc(labels,scores):
    assert auroc(labels,scores)['value'] is None
    assert auroc(labels,scores)['reason']

@pytest.mark.parametrize('labels,scores', [([2],[1]),([0],[float('nan')]),([1,0],[1]),([[0]],[1])])
def test_invalid(labels,scores):
    with pytest.raises(ValidationError): auroc(labels,scores)

def test_strict_threshold_and_counts():
    result = operating_metrics([0,0,1,1],[.5,.6,.5,.9],.5)
    assert [result[k] for k in ['tp','fn','fp','tn']] == [1,1,1,1]
    assert result['defect_recall']['value'] == .5
    assert result['false_positive_rate']['value'] == .5
    assert operating_metrics([1],[1],.5)['false_positive_rate']['value'] is None
    assert operating_metrics([0,1],[0,1],float('inf'))['tp'] is None

def test_macro_equal_image_weight_and_exclusion():
    rows = [{'pixel_auroc': {'value': v}, 'positive_pixels': n, 'negative_pixels': n} for v,n in [(1,1000),(0,1),(None,0)]]
    result = pixel_macro(rows)
    assert result['auroc']['value'] == .5
    assert result['eligible_images'] == 2 and result['excluded_images'] == 1
    assert pixel_macro([])['auroc']['value'] is None

def test_latency():
    assert latency_summary([1,2,3])['p50_ms'] == 2
    assert latency_summary([])['p50_ms'] is None
    with pytest.raises(ValidationError): latency_summary([-1])
