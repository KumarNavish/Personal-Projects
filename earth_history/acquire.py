"""Read two annual representations on the same grid. No risk or event-causation claim."""
from pathlib import Path
import io,json,time,hashlib,traceback
import requests,pandas as pd,numpy as np,rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
OUT=Path('earth_history_output');OUT.mkdir(exist_ok=True)
B=[450560,5235200,491520,5276160];W=H=1024;RES=40;YEAR=2017
R={'status':'STARTING','year':YEAR,'bounds':B,'crs':'EPSG:32632','width':W,'height':H,'resolution_m':RES,'sources':[],'claim':'Recorded 2017 representation on a matching grid. Temporal vector differences are not proof of construction, loss, or an identified physical cause.','attribution':'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.'}
def save():
 (OUT/'manifest.json').write_text(json.dumps(R,indent=2,default=str));print(R['status'],flush=True)
t=time.monotonic();save()
try:
 r=requests.get('https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet',timeout=60);r.raise_for_status()
 df=pd.read_parquet(io.BytesIO(r.content));R['index_sha256']=hashlib.sha256(r.content).hexdigest()
 rows=df[(df.year==YEAR)&(df.crs=='EPSG:32632')&(df.utm_west<B[2])&(df.utm_east>B[0])&(df.utm_south<B[3])&(df.utm_north>B[1])]
 assert len(rows)==2,len(rows)
 raw=np.full((64,H,W),-128,dtype=np.int8)
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_TIMEOUT='90',GDAL_HTTP_MAX_RETRY='1',GDAL_CACHEMAX=128):
  for path in rows.path:
   url=path.replace('gs://','https://storage.googleapis.com/')
   with rasterio.open(url) as ds:
    assert ds.count==64 and ds.crs.to_epsg()==32632
    assert ds.transform.b==ds.transform.d==0 and abs(ds.transform.a)==abs(ds.transform.e)==10
    assert 4 in ds.overviews(1)
    xx=sorted([ds.bounds.left,ds.bounds.right]);yy=sorted([ds.bounds.bottom,ds.bounds.top])
    left=max(B[0],xx[0]);right=min(B[2],xx[1]);bottom=max(B[1],yy[0]);top=min(B[3],yy[1])
    corners=[(~ds.transform)*(x,y) for x in [left,right] for y in [bottom,top]];cols,rs=zip(*corners)
    win=Window(min(cols),min(rs),max(cols)-min(cols),max(rs)-min(rs))
    assert all(abs(v/4-round(v/4))<1e-7 for v in [win.col_off,win.row_off,win.width,win.height])
    w=int((right-left)/RES);h=int((top-bottom)/RES)
    a=ds.read(window=win,out_shape=(64,h,w),resampling=Resampling.nearest)
    if ds.transform.e>0:a=a[:,::-1,:]
    if ds.transform.a<0:a=a[:,:,::-1]
    row=int((B[3]-top)/RES);col=int((left-B[0])/RES);raw[:,row:row+h,col:col+w]=a
    R['sources'].append({'url':url,'window':[win.col_off,win.row_off,win.width,win.height],'transform':list(ds.transform),'flip_y':ds.transform.e>0})
   save()
 a=raw.transpose(1,2,0).copy();valid=(a!=-128).all(-1);assert valid.mean()>.95
 a.tofile(OUT/'embeddings_2017.int8')
 R.update(status='COMPLETE',valid_fraction=float(valid.mean()),cells=int(valid.sum()),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest(),seconds=time.monotonic()-t)
 save()
except Exception as e:
 R.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
