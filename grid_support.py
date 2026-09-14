"""Compatibility only: modern import name and NumPy NaN spelling.
No policy, action-set, forecast, reward, topology or solver equation changes.
"""
from pathlib import Path
import hashlib,json,runpy,subprocess
src=Path('/tmp/ljn-public');out=Path('grid_output');out.mkdir(exist_ok=True)
patches=[]
for p in src.rglob('*.py'):
 old=p.read_text();new=old.replace('np.NaN','np.nan')
 if p.name=='convex_optim.py':
  new=new.replace('from lightsim2grid.gridmodel import init\n','try:\n    from lightsim2grid.gridmodel import init_from_pandapower as init\nexcept ImportError:\n    from lightsim2grid.network import init_from_pandapower as init\n')
 if old!=new:
  patches.append({'file':str(p.relative_to(src)),'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(new.encode()).hexdigest()});p.write_text(new)
(out/'compatibility.json').write_text(json.dumps({'prior_failed_run':34842984048,'patches':patches,'scope':'Import aliases and NaN symbol only. Not a performance-method modification.'},indent=2))
(out/'upstream-compatibility.diff').write_text(subprocess.check_output(['git','-C',str(src),'diff'],text=True))
runpy.run_path('grid_probe/run.py',run_name='__main__')
