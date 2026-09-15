"""Streaming public-benchmark simulation. No connection to any physical grid.
The endpoint runs DoNothing and the released LJN controller from the same chronic
and streams actual simulator states. One run at a time by design.
"""
import os,sys,json,time,threading,types,subprocess,hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import numpy as np
import torch,grid2op
from lightsim2grid import LightSimBackend
from grid2op.Agent import DoNothingAgent
SRC=os.environ.get('LJN_SOURCE','/tmp/ljn-public')
src=__import__('pathlib').Path(SRC);pkg=types.ModuleType('ljn_public');pkg.__path__=[str(src)];sys.modules['ljn_public']=pkg
p=src/'make_agent.py';s=p.read_text();p.write_text(s.replace('from .LJNagent import','from .LJNAgent import'))
from ljn_public.make_agent import make_agent_topoNN
from ljn_public.modules.rewards import MaxRhoReward
torch.set_num_threads(int(os.environ.get('TORCH_THREADS','2')))
COMMIT=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip();POLICY_SHA=hashlib.sha256((src/'models/RL_training_PPO.zip').read_bytes()).hexdigest();LOCK=threading.Lock()
def make_env(ep):
 e=grid2op.make('l2rpn_idf_2023',test=True,backend=LightSimBackend(),reward_class=MaxRhoReward);e.seed(20260914);e.set_id(ep);o=e.reset();return e,o
def view(obs):return {'step':int(obs.current_step),'time':str(obs.get_time_stamp()),'maxRho':float(np.nanmax(obs.rho)),'rho':[round(float(x),4) for x in obs.rho],'lineStatus':[bool(x) for x in obs.line_status]}
def event(ep):
 ew,ow=make_env(ep);ea,oa=make_env(ep);aw=DoNothingAgent(ew.action_space);aa=make_agent_topoNN(ea,str(src));aa.seed(20260914);aa.reset(oa);rw=ra=0.;dw=da=False;start=time.monotonic();actions=0
 try:
  yield {'type':'start','episode':ep,'runStartedUnix':time.time(),'controllerCommit':COMMIT,'policySha256':POLICY_SHA,'substations':ea.n_sub,'powerlines':ea.n_line,'loads':ea.n_load,'generators':ea.n_gen}
  for _ in range(576):
   w_before=view(ow) if not dw else None;a_before=view(oa) if not da else None;ad={}
   if not dw:
    ow,rw,dw,iw=ew.step(aw.act(ow,rw,dw))
   else:iw={}
   if not da:
    act=aa.act(oa,ra,da);ad=act.as_dict();actions+=int(bool(ad));oa,ra,da,ia=ea.step(act)
   else:ia={}
   yield {'type':'step','episode':ep,'step':max(int(ow.current_step) if not dw else 0,int(oa.current_step) if not da else 0),'wait':{'before':w_before,'done':dw,'exceptions':[str(x) for x in iw.get('exception',[])]},'act':{'before':a_before,'action':ad,'done':da,'exceptions':[str(x) for x in ia.get('exception',[])]},'elapsedSeconds':time.monotonic()-start,'actionCount':actions}
   if dw and da:break
  yield {'type':'end','episode':ep,'wait':view(ow),'act':view(oa),'waitDone':dw,'actDone':da,'actionCount':actions,'elapsedSeconds':time.monotonic()-start}
 finally:ew.close();ea.close()
class H(BaseHTTPRequestHandler):
 def _headers(self,code=200,ctype='application/json'):
  self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Access-Control-Allow-Origin','*');self.end_headers()
 def log_message(self,fmt,*args):print('HTTP',self.address_string(),fmt%args,flush=True)
 def do_GET(self):
  u=urlparse(self.path)
  if u.path=='/api/health':self._headers();self.wfile.write(json.dumps({'ok':True,'scope':'public research simulation only','controllerCommit':COMMIT,'policySha256':POLICY_SHA}).encode());return
  if u.path!='/api/run':self._headers(404);self.wfile.write(b'{}');return
  q=parse_qs(u.query);ep=int(q.get('episode',['0'])[0]);
  if ep not in (0,1):self._headers(400);self.wfile.write(b'{"error":"only bundled test episodes 0 and 1"}');return
  if not LOCK.acquire(blocking=False):self._headers(409);self.wfile.write(b'{"error":"run already active"}');return
  try:
   self._headers(200,'application/x-ndjson')
   for row in event(ep):self.wfile.write((json.dumps(row,separators=(',',':'))+'\n').encode());self.wfile.flush()
  except (BrokenPipeError,ConnectionResetError):pass
  finally:LOCK.release()
if __name__=='__main__':ThreadingHTTPServer(('0.0.0.0',int(os.environ.get('PORT','8787'))),H).serve_forever()
