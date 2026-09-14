"""Mechanical policy export only: no retraining, no new action set or altered grid."""
from pathlib import Path
import sys,types,json,hashlib,time
import numpy as np, torch,grid2op
from lightsim2grid import LightSimBackend
R=Path('grid_runtime_output');R.mkdir(exist_ok=True)
src=Path('/tmp/ljn-public')
pkg=types.ModuleType('ljn_public');pkg.__path__=[str(src)];sys.modules['ljn_public']=pkg
p=src/'make_agent.py';s=p.read_text();p.write_text(s.replace('from .LJNagent import','from .LJNAgent import'))
from ljn_public.make_agent import make_agent_topoNN
from ljn_public.modules.rewards import MaxRhoReward
torch.set_num_threads(2)
env=grid2op.make('l2rpn_idf_2023',test=True,backend=LightSimBackend(),reward_class=MaxRhoReward)
env.seed(20260914);env.set_id(0);obs=env.reset();agent=make_agent_topoNN(env,str(src));policy=agent.topo_12_unsafe.model.policy
assert type(policy.features_extractor).__name__=='FlattenExtractor'
weights={};layers=[]
for i,layer in enumerate(list(policy.mlp_extractor.policy_net)+[policy.action_net]):
 name=type(layer).__name__
 if name=='Linear':
  weights[f'w{i}']=layer.weight.detach().cpu().numpy();weights[f'b{i}']=layer.bias.detach().cpu().numpy();layers.append({'op':'linear','w':f'w{i}','b':f'b{i}'})
 elif name in ('Tanh','ReLU','ELU'):layers.append({'op':name.lower()})
 else:raise ValueError('Unsupported actor operation: '+name)
np.savez_compressed(R/'actor.npz',**weights)
(R/'actor.json').write_text(json.dumps(layers,indent=2))
def forward(x):
 y=np.array(x,dtype=np.float32,copy=True)
 for layer in layers:
  op=layer['op']
  if op=='linear':y=y@weights[layer['w']].T+weights[layer['b']]
  elif op=='tanh':y=np.tanh(y)
  elif op=='relu':y=np.maximum(y,0)
  elif op=='elu':y=np.where(y>=0,y,np.expm1(y))
 return y
checks=[];rng=np.random.default_rng(20260914)
inputs=[rng.uniform(0,2.5,(186,)).astype(np.float32) for _ in range(128)]
# Actual observations, not just generated vectors. Inputs are never fitted.
for step in range(128):
 inputs.append(agent.topo_12_unsafe.gym_env.observation_space.to_gym(obs))
 a=agent.act(obs,0,False);obs,_,done,_=env.step(a)
 if done:break
for x in inputs:
 with torch.no_grad():
  z=torch.from_numpy(x).reshape(1,-1)
  f=policy.extract_features(z)
  t=policy.action_net(policy.mlp_extractor.forward_actor(f)).cpu().numpy()[0]
  ids=torch.topk(torch.from_numpy(t),20)[1].numpy()
 n=forward(x);nid=np.argsort(-n,kind='stable')[:20]
 checks.append({'max_abs_logit_error':float(np.max(np.abs(t-n))),'same_top20_order':bool(np.array_equal(ids,nid))})
assert all(c['same_top20_order'] for c in checks),'Export changes tested candidate order'
manifest={'status':'VERIFIED_EXPORT','tests':len(checks),'same_top20_all':True,'max_abs_logit_error':max(c['max_abs_logit_error'] for c in checks),'layers':layers,'source_policy_sha256':hashlib.sha256((src/'models/RL_training_PPO.zip').read_bytes()).hexdigest(),'export_sha256':hashlib.sha256((R/'actor.npz').read_bytes()).hexdigest(),'scope':'Equivalence of top-20 ordering on 128 random load-fraction vectors and up to 128 actual observations, not a formal all-input equivalence proof. Full trajectory check follows before deployment.'}
(R/'export-verification.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
env.close()
