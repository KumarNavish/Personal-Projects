"""Read public Earth-observation metadata. No credentials, private data or inference service."""
from pathlib import Path
import requests,pandas as pd,json,hashlib,time
root=Path('earth_output');root.mkdir(exist_ok=True)
url='https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet'
t=time.monotonic();r=requests.get(url,timeout=60);r.raise_for_status()
assert len(r.content)<100_000_000,'Unexpectedly large index'
p=root/'index.parquet';p.write_bytes(r.content);df=pd.read_parquet(p)
rows=df[(df.year==2025)&(df.wgs84_west<8.8)&(df.wgs84_east>8.4)&(df.wgs84_south<47.6)&(df.wgs84_north>47.2)]
# GeoParquet geometry is WKB, not UTF-8. Omit it; explicit numeric bounds are preserved.
records=rows.drop(columns=['geometry'],errors='ignore').to_dict(orient='records')
result={'status':'INDEX_READ','source':url,'index_sha256':hashlib.sha256(r.content).hexdigest(),'columns':list(df.columns),'matching_rows':records,'elapsed_seconds':time.monotonic()-t,'attribution':'The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.','license':'CC BY 4.0','geometry_or_risk_claim':False,'previous_failed_run':34837480391,'fix':'Do not cast binary WKB geometry to UTF-8'}
text=json.dumps(result,indent=2,default=lambda v:v.hex() if isinstance(v,bytes) else str(v))
(root/'index-result.json').write_text(text);print(text,flush=True)
