"""Independent visual checks of all eight returned candidates per executed query.
These images never enter the AlphaEarth retrieval. No field truth, class detection,
acquisition-year identity, customer data, or property-risk claim is made.
"""
from pathlib import Path
import requests,json,time,hashlib,io
from PIL import Image
OUT=Path('earth_detail_output');OUT.mkdir(exist_ok=True)
POSITIONS=[[608,99],[399,689],[689,303],[359,375],[550,416],[627,367],[439,274],[534,217],[623,513],[705,181],[21,389],[463,59],[790,501],[345,760],[431,738],[544,488],[344,428],[320,727],[890,438],[792,410],[826,358],[933,383],[832,439],[365,644],[952,476],[400,678],[628,423],[406,440],[600,603],[691,439],[593,65],[829,113],[452,640],[1004,676],[300,623],[308,794],[919,424],[984,484],[959,454],[1021,492],[861,400],[822,390],[1002,451],[1023,532],[751,604]]
R={'status':'STARTING','attribution':'Source: Federal Office of Topography swisstopo','selection':'All eight automatically returned candidates for five previously executed, fixed geographic queries. No candidate selected by high-resolution appearance.','input_vector_sha256':'a9ef92247916e12201267223226aa330b55dacb6dd7da7bc1272a15edf9cf776','crs':'EPSG:32632','footprint_m':1200,'image_pixels':640,'date_rule':'Current provider mosaic with differing source acquisition dates. Not necessarily 2025 or live.','images':[]}
def save():
 (OUT/'detail-manifest.json').write_text(json.dumps(R,indent=2));print(R['status'],len(R['images']),flush=True)
t=time.monotonic();save()
for row,col in POSITIONS:
 x=450560+(col+.5)*40;y=5276160-(row+.5)*40
 params={'SERVICE':'WMS','VERSION':'1.3.0','REQUEST':'GetMap','LAYERS':'ch.swisstopo.swissimage','STYLES':'default','CRS':'EPSG:32632','BBOX':f'{x-600},{y-600},{x+600},{y+600}','WIDTH':640,'HEIGHT':640,'FORMAT':'image/jpeg'}
 response=requests.get('https://wms.geo.admin.ch/',params=params,timeout=45);response.raise_for_status();raw=response.content
 image=Image.open(io.BytesIO(raw));image.load();assert image.size==(640,640)
 filename=f'{row}-{col}.jpg';(OUT/filename).write_bytes(raw)
 R['images'].append({'row':row,'col':col,'file':filename,'url':response.url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});R['status']='PARTIAL';save();time.sleep(.3)
R.update(status='COMPLETE',total_seconds=time.monotonic()-t);save()
