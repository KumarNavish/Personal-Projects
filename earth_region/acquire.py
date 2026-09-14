"""Read authentic public learned vectors; explicit south-up to north-up coordinate handling."""
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np,rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
OUT=Path('earth_region_output');OUT.mkdir(exist_ok=True)
B=[450560,5235200,491520,5276160];RES=40;W=H=1024
SOURCES=[
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xi95ukhtjl2axdx22-0000000000-0000008192.tiff',
 'https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/2025/32N/xr92gz3q6qy129tf2-0000008192-0000008192.tiff']
R={'status':'STARTING','year':2025,'crs':'EPSG:32632','bounds':B,'width':W,'height':H,'resolution_m':RES,'source_resolution_m':10,'overview':'Provider normalized-mean native 4x overview, nearest sampling, exact grid alignment','scope':'Landscape-vector similarity. Not risk, ownership, inventory, forecast or physical survey.','sources':[],'human_impact':'UNVALIDATED','attribution':'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prior_empty_acquisition_run':34837991815,'fix':'Provider rasters have a positive Y pixel step (south-up). Sort geographic bounds, derive pixel windows through inverse transform, then flip read arrays into north-up output.','reused_aerial_artifact':10344018795}
def save():
 (OUT/'manifest.json').write_text(json.dumps(R,indent=2,default=str));print(json.dumps(R),flush=True)
t=time.monotonic();save()
try:
 mosaic=np.full((64,H,W),-128,dtype=np.int8)
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_TIMEOUT='90',GDAL_HTTP_MAX_RETRY='1',GDAL_CACHEMAX=128):
  for url in SOURCES:
   st=time.monotonic()
   with rasterio.open(url) as ds:
    assert ds.count==64 and ds.crs.to_epsg()==32632
    assert abs(ds.transform.a)==10 and abs(ds.transform.e)==10 and ds.transform.b==ds.transform.d==0
    assert 4 in ds.overviews(1)
    xb=sorted([ds.bounds.left,ds.bounds.right]);yb=sorted([ds.bounds.bottom,ds.bounds.top])
    left=max(B[0],xb[0]);bottom=max(B[1],yb[0]);right=min(B[2],xb[1]);top=min(B[3],yb[1])
    if right<=left or top<=bottom:raise ValueError('Indexed source has no geographic intersection')
    corners=[(~ds.transform)*(x,y) for x in [left,right] for y in [bottom,top]]
    cols,rows=zip(*corners)
    win=Window(min(cols),min(rows),max(cols)-min(cols),max(rows)-min(rows))
    assert all(abs(v/4-round(v/4))<1e-7 for v in [win.col_off,win.row_off,win.width,win.height]),'Not aligned to native overview'
    w=int((right-left)/RES);h=int((top-bottom)/RES)
    arr=ds.read(window=win,out_shape=(64,h,w),resampling=Resampling.nearest)
    assert arr.dtype==np.int8
    if ds.transform.e>0:arr=arr[:,::-1,:]
    if ds.transform.a<0:arr=arr[:,:,::-1]
    row=int((B[3]-top)/RES);col=int((left-B[0])/RES)
    mosaic[:,row:row+h,col:col+w]=arr
    R['sources'].append({'url':url,'shape':arr.shape,'native_shape':[ds.height,ds.width],'transform':list(ds.transform),'native_bounds':list(ds.bounds),'read_window':[win.col_off,win.row_off,win.width,win.height],'output_row_col':[row,col],'flipped_y':ds.transform.e>0,'elapsed_seconds':time.monotonic()-st})
   R['status']='PARTIAL_RASTER';save()
 raw=mosaic.transpose(1,2,0).copy();valid=(raw!=-128).all(-1)
 assert valid.mean()>.95,'Insufficient source coverage; no success result'
 samples=raw.reshape(-1,64)[::83];unique=len(np.unique(samples,axis=0))
 assert unique>1000,'Unexpectedly constant data'
 raw.tofile(OUT/'embeddings.int8')
 R.update(status='COMPLETE',cells=int(valid.sum()),valid_fraction=float(valid.mean()),raw_bytes=raw.nbytes,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),sampled_unique_vectors=unique,total_seconds=time.monotonic()-t)
 save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),total_seconds=time.monotonic()-t);save();raise
