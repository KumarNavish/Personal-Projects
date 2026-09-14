"""Read native 10m vectors at a fixed 40m lattice, not normalized-mean overviews.
This samples 1/16 of base pixels. No claim of exhaustive building detection is made.
"""
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np,rasterio
from rasterio.windows import Window
OUT=Path('earth_native_output');OUT.mkdir(exist_ok=True)
B=[450560,5235200,491520,5276160];STRIDE=4;W=H=1024
SOURCES=[
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xi95ukhtjl2axdx22-0000000000-0000008192.tiff',
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xr92gz3q6qy129tf2-0000008192-0000008192.tiff']
R={'status':'STARTING','year':2025,'crs':'EPSG:32632','bounds':B,'width':W,'height':H,'resolution_m':40,'source_resolution_m':10,'sample_offset_m':5,'sampling':'One native 10m source pixel every 40m horizontally and vertically; 1/16 base pixels. No overviews, averaging, interpolation or semantic labels.','scope':'Sampled landscape-similarity retrieval only. Not inventory, coverage of every building, risk, forecast, ownership or survey.','sources':[],'human_impact':'UNVALIDATED','attribution':'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
def save():
 (OUT/'manifest.json').write_text(json.dumps(R,indent=2,default=str));print(R['status'],len(R['sources']),flush=True)
t=time.monotonic();save()
try:
 mosaic=np.full((64,H,W),-128,dtype=np.int8)
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_TIMEOUT='90',GDAL_HTTP_MAX_RETRY='1',GDAL_CACHEMAX=256*1024*1024):
  for url in SOURCES:
   st=time.monotonic()
   with rasterio.open(url) as ds:
    assert ds.count==64 and ds.crs.to_epsg()==32632 and ds.transform.a==ds.transform.e==10
    xb=sorted([ds.bounds.left,ds.bounds.right]);yb=sorted([ds.bounds.bottom,ds.bounds.top])
    left=max(B[0],xb[0]);bottom=max(B[1],yb[0]);right=min(B[2],xb[1]);top=min(B[3],yb[1]);assert right>left and top>bottom
    for north in range(int(top),int(bottom),-5120):
     south=max(bottom,north-5120)
     col,row=(~ds.transform)*(left,south)
     width=int((right-left)/10);height=int((north-south)/10)
     assert all(v%4==0 for v in [int(col),int(row),width,height])
     # No out_shape means a genuine native-resolution read, not GDAL's automatic overview.
     full=ds.read(window=Window(int(col),int(row),width,height))
     assert full.shape==(64,height,width) and full.dtype==np.int8
     north_up=full[:,::-1,:]
     sampled=north_up[:,::4,::4]
     r0=int((B[3]-north)/40);c0=int((left-B[0])/40)
     mosaic[:,r0:r0+sampled.shape[1],c0:c0+sampled.shape[2]]=sampled
     print('NATIVE_STRIP',r0,sampled.shape,flush=True)
    R['sources'].append({'url':url,'transform':list(ds.transform),'native_bounds':list(ds.bounds),'base_resolution_read':True,'elapsed_seconds':time.monotonic()-st});save()
 raw=mosaic.transpose(1,2,0).copy();valid=(raw!=-128).all(-1);assert valid.mean()>.95
 assert len(np.unique(raw.reshape(-1,64)[::83],axis=0))>1000
 raw.tofile(OUT/'embeddings.int8')
 R.update(status='COMPLETE',cells=int(valid.sum()),valid_fraction=float(valid.mean()),raw_bytes=raw.nbytes,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),total_seconds=time.monotonic()-t)
 save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),total_seconds=time.monotonic()-t);save();raise
