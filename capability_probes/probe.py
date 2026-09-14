"""Operate published capabilities against public inputs; no external control or paid API."""
import os, time, json, traceback, importlib.metadata, hashlib, inspect
from pathlib import Path

OUT = Path('probe_output')
OUT.mkdir(exist_ok=True)


def emit(name, value):
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, default=str)
    (OUT / (name + '.json')).write_text(text)
    print('\nRESULT_BEGIN ' + name + '\n' + text + '\nRESULT_END ' + name, flush=True)


def spatial_access():
    import requests
    from gradio_client import Client
    start = time.monotonic()
    space = 'gagndeep/Apple-Sharp-Image-to-3D-View-Synthesis'
    result = {'space': space, 'task': 'Discover the executable interface before inference', 'inference_run': False}
    for name, url in [('space_source', 'https://huggingface.co/spaces/' + space + '/raw/main/app.py'), ('runtime', 'https://huggingface.co/api/spaces/' + space + '/runtime')]:
        try:
            r = requests.get(url, timeout=35)
            result[name] = {'status': r.status_code, 'body': r.text[:22000]}
        except Exception as e:
            result[name] = {'error': str(e)}
    try:
        client = Client(space, verbose=False, httpx_kwargs={'timeout': 40})
        result['api'] = client.view_api(return_format='dict', print_info=False)
    except Exception as e:
        result['api_error'] = repr(e)
    result['elapsed_seconds'] = time.monotonic() - start
    emit('spatial_access', result)


def wind_control():
    import numpy as np
    from scipy.optimize import differential_evolution
    from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site, wt_x, wt_y
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    from py_wake.deflection_models import JimenezWakeDeflection
    start = time.monotonic()
    site, turbines = Hornsrev1Site(), V80()
    model = Bastankhah_PorteAgel_2014(site, turbines, k=0.0324555, deflectionModel=JimenezWakeDeflection())
    x, y = np.array(wt_x, dtype=float), np.array(wt_y, dtype=float)
    assert len(x) == 80 and len(y) == 80
    # Use the published complete farm geometry unchanged. Ten physical columns each share a yaw command.
    groups = np.arange(80) // 8
    directions = [255, 260, 265, 270, 275, 280, 285]
    speed = 8.0
    def powers(yaw, wd=270):
        sim = model(x, y, wd=[wd], ws=[speed], yaw=np.asarray(yaw)[:, None, None], tilt=0)
        return np.asarray(sim.Power).reshape(80)
    baseline = powers(np.zeros(80))
    evaluations = []
    def objective(p):
        value = float(powers(np.asarray(p)[groups]).sum())
        evaluations.append({'yaw_by_column': np.asarray(p).tolist(), 'power_W': value, 'elapsed': time.monotonic()-start})
        return -value
    # Bounded derivative-free optimization, not an assertion of novel control or physical performance.
    optimum = differential_evolution(objective, [(-25, 25)] * 10, maxiter=18, popsize=5, seed=20260914, polish=False, workers=1, x0=np.zeros(10))
    yaw = optimum.x[groups]
    candidate = powers(yaw)
    rows=[]
    for wd in directions:
        a, b = powers(np.zeros(80), wd), powers(yaw, wd)
        rows.append({'direction_deg': wd, 'speed_m_s': speed, 'baseline_W': float(a.sum()), 'candidate_W':float(b.sum()), 'relative_change':float(b.sum()/a.sum()-1)})
    result={'status':'EXECUTED', 'scope':'Steady-state engineering simulation only; no physical turbine commands, field measurements, forecasts or annual-energy claims.', 'layout_source':'DTU PyWake bundled Hornsrev1 dataset', 'model':'Bastankhah_PorteAgel_2014 k=0.0324555 plus JimenezWakeDeflection', 'optimizer':'scipy differential_evolution seed=20260914 maxiter=18 popsize=5 polish=False', 'packages':{p:importlib.metadata.version(p) for p in ['py-wake','scipy','numpy']}, 'turbines':80, 'geometry':np.column_stack([x,y]).tolist(), 'objective_direction_deg':270, 'objective_speed_m_s':speed, 'baseline_W':float(baseline.sum()), 'candidate_W':float(candidate.sum()), 'baseline_per_turbine_W':baseline.tolist(), 'candidate_per_turbine_W':candidate.tolist(), 'yaw_deg':yaw.tolist(), 'direction_sensitivity':rows, 'evaluations':len(evaluations), 'optimizer_termination':str(optimum.message), 'elapsed_seconds':time.monotonic()-start, 'human_effect':'NOT_TESTED'}
    emit('wind_control',result)
    emit('wind_search_trace',evaluations)


if __name__=='__main__':
    name=os.environ.get('PROBE','spatial')
    try:
        {'spatial':spatial_access,'wind':wind_control}[name]()
    except Exception as e:
        emit(name+'_failure',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()})
        raise
