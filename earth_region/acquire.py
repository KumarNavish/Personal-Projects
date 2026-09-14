"""Public learned landscape representations; no customer data or property-risk claims."""
from pathlib import Path
import json,time,hashlib,traceback,io,xml.etree.ElementTree as ET
import numpy as np, requests, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from PIL import Image
OUT=Path('earth_region_output');OUT.mkdir(exist_ok=True)
B=[450560,5235200,491520,5276160];RES=40;W=H=1024
SOURCES=[
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xi95ukhtjl2axdx22-0000000000-0000008192.tiff',
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xr92gz3q6qy129tf2-0000008192-0000008192.tiff']
R={'status':'STARTING','year':2025,'crs':'EPSG:32632','bounds':B,'width':W,'height':H,'resolution_m':RES,'source_resolution_m':10,'overview':'Existing provider-generated normalized mean overview; nearest sampling at exact 4x grid alignment','scope':'Landscape-vector similarity only. Not an asset inventory, physical survey, risk estimate, forecast, ownership inference or detection guarantee.','sources':[],'human_impact':'UNVALIDATED','attribution':'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.','imagery_attribution':'Source: Federal Office of Topography swisstopo','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():
 (OUT/'manifest.json').write_text(json.dumps(R,indent=2,default=str));print(R['status'],flush=True)
t=time.monotonic();save()
try:
 mosaic=np.full((64,H,W),-128,dtype=np.int8)
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_TIMEOUT='60',GDAL_HTTP_MAX_RETRY='1',GDAL_CACHEMAX=128):
  for url in SOURCES:
   st=time.monotonic()
   with rasterio.open(url) as ds:
    assert ds.count==64 and ds.crs.to_epsg()==32632
    assert 4 in ds.overviews(1)
    left=max(B[0],ds.bounds.left);bottom=max(B[1],ds.bounds.bottom);right=min(B[2],ds.bounds.right);top=min(B[3],ds.bounds.top)
    if right<=left or top<=bottom:continue
    w=int((right-left)/RES);h=int((top-bottom)/RES)
    arr=ds.read(window=from_bounds(left,bottom,right,top,ds.transform),out_shape=(64,h,w),resampling=Resampling.nearest)
    assert arr.dtype==np.int8,arr.dtype
    row=int((B[3]-top)/RES);col=int((left-B[0])/RES)
    mosaic[:,row:row+h,col:col+w]=arr
    R['sources'].append({'url':url,'shape':arr.shape,'native_shape':[ds.height,ds.width],'transform':list(ds.transform),'overviews':ds.overviews(1),'bounds':list(ds.bounds),'elapsed_seconds':time.monotonic()-st})
   R['status']='PARTIAL_RASTER';save()
 raw=mosaic.transpose(1,2,0).copy();raw.tofile(OUT/'embeddings.int8')
 valid=(raw!=-128).all(-1)
 R.update(status='RASTER_COMPLETE',cells=int(valid.sum()),valid_fraction=float(valid.mean()),raw_bytes=raw.nbytes,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),raster_seconds=time.monotonic()-t)
 save()
 # Check actual WMS advertised layers before requesting any imagery.
 response=requests.get('https://wms.geo.admin.ch/',params={'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetCapabilities','LANG':'en'},timeout=45);response.raise_for_status()
 root=ET.fromstring(response.content);names={e.text for e in root.iter() if e.tag.split('}')[-1]=='Name'}
 assert 'ch.swisstopo.swissimage' in names,'Swiss aerial layer not advertised'
 R['wms_capabilities_sha256']=hashlib.sha256(response.content).hexdigest()
 # 16 source tiles, regular grid. They are independent visual inspection, not input to similarity.
 big=Image.new('RGB',(4096,4096));R['imagery']=[]
 for row in range(4):
  for col in range(4):
   bb=[B[0]+col*10240,B[3]-(row+1)*10240,B[0]+(col+1)*10240,B[3]-row*10240]
   params={'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetMap','LAYERS':'ch.swisstopo.swissimage','STYLES':'default','CRS':'EPSG:32632','BBOX':','.join(map(str,bb)),'WIDTH':1024,'HEIGHT':1024,'FORMAT':'image/jpeg'}
   r=requests.get('https://wms.geo.admin.ch/',params=params,timeout=60);r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.load();assert im.size==(1024,1024)
   big.paste(im,(col*1024,row*1024));R['imagery'].append({'row':row,'col':col,'bbox':bb,'url':r.url,'sha256':hashlib.sha256(r.content).hexdigest()});time.sleep(.25)
 big.save(OUT/'aerial.jpg',quality=92)
 R.update(status='COMPLETE',imagery_date='Provider current mosaic: acquisition dates vary; not represented as contemporaneous with 2025 embeddings.',imagery_sha256=hashlib.sha256((OUT/'aerial.jpg').read_bytes()).hexdigest(),total_seconds=time.monotonic()-t)
 save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),total_seconds=time.monotonic()-t);save();raise
