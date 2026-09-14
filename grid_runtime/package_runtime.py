from pathlib import Path
import os,shutil,json,subprocess,hashlib,tarfile
R=Path('grid_runtime_output').resolve();src=Path('/tmp/ljn-public');dest=R/'ljn_public';dest.mkdir(exist_ok=True)
for p in src.rglob('*'):
 rel=p.relative_to(src)
 if '.git' in rel.parts or '__pycache__' in rel.parts:continue
 if p.is_file() and (p.suffix=='.py' or p.name in ['LICENSE','LICENSE.txt','AUTHORS.txt','README.md'] or ('assets' in rel.parts and p.suffix=='.npz')):
  q=dest/rel;q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
(dest/'models').mkdir(exist_ok=True)
for name in ['actor.npz','actor.json']:shutil.copy2(R/name,dest/'models'/name)
p=dest/'LJNAgent.py';old=p.read_text();assert old.count('from stable_baselines3 import PPO')==1;p.write_text(old.replace('from stable_baselines3 import PPO',''))
p=dest/'modules/topology_nn_policy.py';shutil.copy2('grid_runtime/numpy_policy.py',p)
# Importing as a package would trigger the training/baseline-only upstream __init__.
(dest/'__init__.py').write_text('# Deployment package. Original modules and MPL license retained.\n')
fonts=[]
for p in R.rglob('*'):
 if p.is_file() and p.suffix.lower() in ['.ttf','.otf','.woff','.woff2','.pfb','.pfm','.ttc']:
  fonts.append(str(p.relative_to(R)));p.unlink()
# Remove only caches; do not remove runtime native libraries or model source.
for p in sorted(R.rglob('__pycache__'),key=lambda x:len(str(x)),reverse=True):shutil.rmtree(p,ignore_errors=True)
record={'purpose':'Only public benchmark simulation. No connection to any power-grid control system.','source_commit':'ca0637eab9f098be7f206ed0e46a3900cd4deec0','transformations':['Case-correct make_agent import','Remove unused PPO import','Replace actor inference with separately checked NumPy evaluation of original weights','Deployment package initializer excludes evaluation/training imports'],'font_files_in_artifact':0,'removed_font_file_count':len(fonts),'source_files':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}}
(R/'runtime-receipt.json').write_text(json.dumps(record,indent=2))
# Preserve executable bits and symlinks for the portable Python distribution.
with tarfile.open('grid-runtime-linux.tar.gz','w:gz',compresslevel=6) as tf:tf.add(R,arcname='grid_runtime')
size=Path('grid-runtime-linux.tar.gz').stat().st_size
assert size<260_000_000,('Unexpected artifact size',size)
print(json.dumps({'artifact':'grid-runtime-linux.tar.gz','bytes':size,'sha256':hashlib.sha256(Path('grid-runtime-linux.tar.gz').read_bytes()).hexdigest(),'font_files':0}),flush=True)
