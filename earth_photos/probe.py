from pathlib import Path
import json,requests,xml.etree.ElementTree as ET,hashlib
from PIL import Image
import io
O=Path('photo_probe_output');O.mkdir(exist_ok=True)
results=[]
for base in ['https://wms.geo.admin.ch/','https://wms.swisstopo.admin.ch/']:
 d={'base':base}
 try:
  r=requests.get(base,params={'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetCapabilities'},timeout=40);r.raise_for_status()
  root=ET.fromstring(r.content)
  layers=[]
  for layer in root.iter():
   if layer.tag.split('}')[-1]!='Layer':continue
   children=list(layer);name=next((x.text for x in children if x.tag.split('}')[-1]=='Name'),None)
   if name and 'swissimage' in name.lower():
    layers.append({'name':name,'xml':ET.tostring(layer,encoding='unicode')[:15000]})
  d['layers']=layers
  if base=='https://wms.swisstopo.admin.ch/' and any(x['name']=='ch.swisstopo.swissimage' for x in layers):
   d['samples']=[]
   for year in ['2017','2025']:
    p={'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetMap','LAYERS':'ch.swisstopo.swissimage','STYLES':'default','CRS':'EPSG:32632','BBOX':'458960,5258320,459960,5259320','WIDTH':512,'HEIGHT':512,'FORMAT':'image/jpeg','TIME':year}
    q=requests.get(base,params=p,timeout=45);q.raise_for_status();im=Image.open(io.BytesIO(q.content));im.load();(O/('sample_'+year+'.jpg')).write_bytes(q.content);d['samples'].append({'year':year,'url':q.url,'sha256':hashlib.sha256(q.content).hexdigest()})
 except Exception as e:d['error']=repr(e)
 results.append(d);(O/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(d),flush=True)
