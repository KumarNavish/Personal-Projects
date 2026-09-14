"""Actual simulation, not a physical-grid or human-operator performance claim.
Operate the authors' released 118-substation controller on the bundled public
IDF2023 chronics. No manual outages, demand edits, weakened safety limits or
selection of successful episodes. Save every action and terminal outcome.
"""
import os,sys,json,time,hashlib,traceback,importlib,types,subprocess
from pathlib import Path
import numpy as np
OUT=Path('grid_output');OUT.mkdir(exist_ok=True)
R={'status':'STARTING','claim':'Run a released controller on a complete 118-substation research grid using unchanged bundled public benchmark time series. This is simulation, not a real French power grid, field prevention, a comparison with human operators, or an assessment of all benchmark scenarios.','human_participants':0,'manual_disturbances':False,'max_steps_per_run':576,'episode_selection':'First two bundled chronics in environment order, before examining any outcome.','seed':20260914,'comparators':['DoNothingAgent','RecoPowerlineAgent','released_LJN_topoNN'],'runs':[],'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
t0=time.monotonic()
def plain(o):
 if isinstance(o,np.ndarray):return o.tolist()
 if isinstance(o,np.generic):return o.item()
 return str(o)
def save():
 (OUT/'result.json').write_text(json.dumps(R,indent=2,default=plain));print('CHECKPOINT',R['status'],len(R['runs']),flush=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snap(obs):
 return {'step':int(obs.current_step),'timestamp':str(obs.get_time_stamp()),'rho':obs.rho.tolist(),'line_status':obs.line_status.tolist(),'load_MW':obs.load_p.tolist(),'generation_MW':obs.gen_p.tolist(),'powerflow_MW':obs.p_or.tolist(),'topology':obs.topo_vect.tolist(),'storage_MW':obs.storage_power.tolist(),'curtailed_MW':obs.curtailment_mw.tolist() if hasattr(obs,'curtailment_mw') else None,'max_rho':float(np.nanmax(obs.rho)),'load_total_MW':float(obs.load_p.sum())}
try:
 import torch,grid2op,cvxpy,stable_baselines3,lightsim2grid
 torch.set_num_threads(2)
 from lightsim2grid import LightSimBackend
 from grid2op.Agent import DoNothingAgent,RecoPowerlineAgent
 # Package wrapper only: upstream __init__ assumes installation as a submodule.
 src=Path('/tmp/ljn-public');pkg=types.ModuleType('ljn_public');pkg.__path__=[str(src)];sys.modules['ljn_public']=pkg
 source_sha=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
 # Linux filename compatibility: preserve original and record the one import correction.
 p=src/'make_agent.py';old=p.read_text();new=old.replace('from .LJNagent import','from .LJNAgent import');p.write_text(new)
 R['upstream']={'repository':'https://github.com/lajavaness/l2rpn-2023-ljn-agent','commit':source_sha,'license':'MPL-2.0','import_only_patch':old!=new,'policy_sha256':digest(src/'models/RL_training_PPO.zip')}
 from ljn_public.make_agent import make_agent_topoNN
 from ljn_public.modules.rewards import MaxRhoReward
 R['versions']={m.__name__:getattr(m,'__version__','unknown') for m in [np,torch,grid2op,cvxpy,stable_baselines3,lightsim2grid]}
 env=grid2op.make('l2rpn_idf_2023',test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
 assert env.n_sub==118 and env.n_line==186,(env.n_sub,env.n_line)
 available=env.chronics_handler.subpaths
 R['chronics']=[str(p) for p in available];R['environment']={'name':env.name,'substations':env.n_sub,'powerlines':env.n_line,'loads':env.n_load,'generators':env.n_gen,'storage':env.n_storage,'path':env.get_path_env(),'parameters':env.parameters.to_dict(),'thermal_limits_A':env.get_thermal_limit().tolist(),'graph_line_from':env.line_or_to_subid.tolist(),'graph_line_to':env.line_ex_to_subid.tolist(),'layout':env.grid_layout}
 R['input_files']={str(p.relative_to(Path(env.get_path_env()))):digest(p) for p in Path(env.get_path_env()).rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.bz2','.info']}
 R['status']='PREPARED';save()
 total_budget=650
 for idx in range(min(2,len(available))):
  for name in R['comparators']:
   if time.monotonic()-t0>total_budget:raise TimeoutError('Bounded experiment budget reached; retain completed runs.')
   env.seed(R['seed']);env.set_id(idx);obs=env.reset()
   if name=='DoNothingAgent':agent=DoNothingAgent(env.action_space)
   elif name=='RecoPowerlineAgent':agent=RecoPowerlineAgent(env.action_space)
   else:agent=make_agent_topoNN(env,str(src))
   agent.seed(R['seed']);agent.reset(obs)
   trace=OUT/f'episode-{idx}-{name}.jsonl';run={'episode':idx,'controller':name,'initial':snap(obs),'status':'RUNNING','steps':0,'changed_actions':0,'illegal_actions':0,'decision_seconds':[],'trace_file':trace.name};R['runs'].append(run);R['status']='RUNNING';save()
   reward=0.;done=False;st=time.monotonic()
   with trace.open('w') as f:
    for step in range(R['max_steps_per_run']):
     if time.monotonic()-t0>total_budget:raise TimeoutError('Bounded experiment budget reached during a recorded run.')
     at=time.monotonic();act=agent.act(obs,reward,done);decision=time.monotonic()-at
     record={'before':snap(obs),'action':act.as_dict(),'action_vector':act.to_vect().tolist(),'decision_seconds':decision}
     obs,reward,done,info=env.step(act)
     record.update(after=snap(obs),reward=float(reward),done=bool(done),illegal=bool(info.get('is_illegal',False)),ambiguous=bool(info.get('is_ambiguous',False)),exceptions=[str(e) for e in info.get('exception',[])],disconnected_lines=np.asarray(info.get('disc_lines',[])).tolist())
     f.write(json.dumps(record,default=plain)+'\n');f.flush()
     run['steps']=step+1;run['decision_seconds'].append(decision);run['changed_actions']+=int(act!=env.action_space());run['illegal_actions']+=int(record['illegal'])
     if done:run['status']='TERMINAL';run['terminal_info']=record;break
     if step%48==0:save()
    if not done:run['status']='HORIZON_COMPLETE'
   run['elapsed_seconds']=time.monotonic()-st;run['trace_sha256']=digest(trace);run['final']=snap(obs);save()
 env.close();R.update(status='COMPLETE',total_seconds=time.monotonic()-t0);save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),total_seconds=time.monotonic()-t0);save();print(R['traceback'],flush=True);raise
