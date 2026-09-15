"""Prospectively selected third benchmark chronic. Same released controller and baselines as prior run. No outcome-based episode selection."""
import sys,json,time,hashlib,traceback,types,subprocess
from pathlib import Path
import numpy as np
OUT=Path('grid_episode2_output');OUT.mkdir(exist_ok=True)
R={'status':'STARTING','episode':2,'selection_rule':'The next chronic in environment order, selected before any outcome was inspected.','claim':'Run the same released autonomous controller and the same two baselines on the third untouched public benchmark chronic. Simulation only; no physical-grid or human-operator claim.','controllers':['DoNothingAgent','RecoPowerlineAgent','released_LJN_topoNN'],'max_steps':576,'runs':[],'human_participants':0}
def plain(o):
 if isinstance(o,np.ndarray):return o.tolist()
 if isinstance(o,np.generic):return o.item()
 return str(o)
def save():
 (OUT/'result.json').write_text(json.dumps(R,indent=2,default=plain));print('CHECKPOINT',R['status'],len(R['runs']),flush=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snap(obs):return {'step':int(obs.current_step),'timestamp':str(obs.get_time_stamp()),'max_rho':float(np.nanmax(obs.rho)),'rho':obs.rho.tolist(),'line_status':obs.line_status.tolist(),'load_total_MW':float(obs.load_p.sum())}
t0=time.monotonic()
try:
 import torch,grid2op,lightsim2grid
 torch.set_num_threads(2)
 from lightsim2grid import LightSimBackend
 from grid2op.Agent import DoNothingAgent,RecoPowerlineAgent
 src=Path('/tmp/ljn-public');pkg=types.ModuleType('ljn_public');pkg.__path__=[str(src)];sys.modules['ljn_public']=pkg
 p=src/'make_agent.py';old=p.read_text();new=old.replace('from .LJNagent import','from .LJNAgent import');p.write_text(new)
 source_sha=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
 from ljn_public.make_agent import make_agent_topoNN
 from ljn_public.modules.rewards import MaxRhoReward
 env=grid2op.make('l2rpn_idf_2023',test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
 assert env.n_sub==118 and env.n_line==186
 available=env.chronics_handler.subpaths;assert len(available)>2
 R['environment']={'name':env.name,'substations':env.n_sub,'powerlines':env.n_line,'loads':env.n_load,'generators':env.n_gen,'chronics':[str(x) for x in available]}
 R['upstream']={'repository':'https://github.com/lajavaness/l2rpn-2023-ljn-agent','commit':source_sha,'policy_sha256':digest(src/'models/RL_training_PPO.zip'),'import_only_patch':old!=new}
 R['status']='PREPARED';save()
 for name in R['controllers']:
  env.seed(20260914);env.set_id(2);obs=env.reset()
  agent=DoNothingAgent(env.action_space) if name=='DoNothingAgent' else RecoPowerlineAgent(env.action_space) if name=='RecoPowerlineAgent' else make_agent_topoNN(env,str(src))
  agent.seed(20260914);agent.reset(obs)
  trace=OUT/f'episode-2-{name}.jsonl';run={'controller':name,'initial':snap(obs),'status':'RUNNING','steps':0,'changed_actions':0,'illegal_actions':0};R['runs'].append(run);save();reward=0.;done=False
  with trace.open('w') as f:
   for step in range(R['max_steps']):
    act=agent.act(obs,reward,done);record={'before':snap(obs),'action':act.as_dict()};obs,reward,done,info=env.step(act);record.update(after=snap(obs),reward=float(reward),done=bool(done),illegal=bool(info.get('is_illegal',False)),exceptions=[str(e) for e in info.get('exception',[])])
    f.write(json.dumps(record,default=plain)+'\n');run['steps']=step+1;run['changed_actions']+=int(bool(act.as_dict()));run['illegal_actions']+=int(record['illegal'])
    if done:run['status']='TERMINAL';run['terminal']=record;break
   if not done:run['status']='HORIZON_COMPLETE'
  run['trace_file']=trace.name;run['trace_sha256']=digest(trace);run['final']=snap(obs);save()
 env.close();R.update(status='COMPLETE',total_seconds=time.monotonic()-t0);save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),total_seconds=time.monotonic()-t0);save();print(R['traceback'],flush=True);raise
