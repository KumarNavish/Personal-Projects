from pathlib import Path
import rasterio,json,requests,pandas as pd,io
OUT=Path('header_output');OUT.mkdir(exist_ok=True)
r=requests.get('https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet',timeout=60);r.raise_for_status();df=pd.read_parquet(io.BytesIO(r.content))
rows=df[(df.year==2025)&(df.utm_zone=='32N')&(df.wgs84_west<8.8)&(df.wgs84_east>8.4)&(df.wgs84_south<47.6)&(df.wgs84_north>47.2)]
records=[]
for row in rows.to_dict('records'):
 url=row['path'].replace('gs://','https://storage.googleapis.com/')
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff',GDAL_HTTP_TIMEOUT='40'):
  with rasterio.open(url) as ds:
   d={'url':url,'index':{k:str(v) for k,v in row.items() if k not in ['geom','geometry','geom_bbox']},'raster':{'bounds':list(ds.bounds),'transform':list(ds.transform),'shape':[ds.height,ds.width],'crs':str(ds.crs),'dtype':ds.dtypes,'overviews':ds.overviews(1),'tags':ds.tags()}}
   records.append(d);print(json.dumps(d),flush=True)
(OUT/'headers.json').write_text(json.dumps(records,indent=2))
