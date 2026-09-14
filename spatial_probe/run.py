"""Use a released spatial model on public ETH photographs, not a physical-truth claim."""
import json,time,hashlib,traceback,subprocess,sys,os
from pathlib import Path
import requests
OUT=Path('spatial_output');OUT.mkdir(exist_ok=True)
record={'status':'STARTING','model':'facebook/map-anything-apache','scope':'Estimated geometry from photographs; not a site survey, full unseen world, physical prediction, safety decision or autonomous action.','images':[],'human_effect':'NOT_TESTED'}
def save():
    (OUT/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False,default=str))
    print('CHECKPOINT',record['status'],flush=True)
def get(url):
    r=requests.get(url,timeout=60,headers={'User-Agent':'SpatialCapabilityProbe/1.0 educational-experiment (public-image research)'})
    r.raise_for_status();return r
try:
    start=time.monotonic()
    # These independent photographs were selected by place, not by model reconstruction performance.
    for filename in ['ETH Z\u00fcrich Haupthalle.jpg','ETH Z\u00fcrich Haupthalle Osten.jpg']:
        api='https://commons.wikimedia.org/w/api.php'
        params={'action':'query','format':'json','titles':'File:'+filename,'prop':'imageinfo','iiprop':'url|extmetadata','iiurlwidth':1024}
        r=requests.get(api,params=params,timeout=45,headers={'User-Agent':'SpatialCapabilityProbe/1.0 (public-image research)'})
        r.raise_for_status();j=r.json();pages=list(j['query']['pages'].values());info=pages[0]['imageinfo'][0]
        url=info.get('thumburl',info['url']);raw=get(url).content
        path=OUT/('input_%d.jpg'%len(record['images']));path.write_bytes(raw)
        record['images'].append({'file':path.name,'source_file':filename,'source_url':info['descriptionurl'],'image_url':url,'metadata':info.get('extmetadata',{}),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    record['status']='IMAGES_FETCHED';save()
    import torch,numpy as np
    torch.set_num_threads(2)
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    record['torch_version']=torch.__version__
    t=time.monotonic();model=MapAnything.from_pretrained('facebook/map-anything-apache').to('cpu').eval()
    record['model_load_seconds']=time.monotonic()-t
    record['parameters']=sum(p.numel() for p in model.parameters())
    record['status']='MODEL_LOADED';save()
    views=load_images([str(OUT/i['file']) for i in record['images']])
    t=time.monotonic()
    with torch.inference_mode():
        predictions=model.infer(views,memory_efficient_inference=False,use_amp=False,apply_mask=True,mask_edges=True,apply_confidence_mask=False)
    record['inference_seconds']=time.monotonic()-t
    def array(t):return t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    pts,cols=[],[];record['views']=[]
    for i,p in enumerate(predictions):
        xyz=array(p['pts3d']).squeeze(0);rgb=array(p['img_no_norm']).squeeze(0)
        mask=array(p['mask']).squeeze().astype(bool)
        valid=mask & np.isfinite(xyz).all(-1)
        # Fixed regular sub-sampling only for an inspectable compact artifact, not altering inference.
        keep=np.zeros_like(valid);keep[::3,::3]=True;valid&=keep
        pts.append(xyz[valid]);cols.append((rgb[valid]*255).clip(0,255).astype(np.uint8))
        record['views'].append({'shape':list(xyz.shape),'valid_fraction':float(mask.mean()),'kept_points':int(valid.sum()),'intrinsics':array(p['intrinsics']).tolist(),'camera_poses':array(p['camera_poses']).tolist()})
    xyz=np.concatenate(pts).astype('<f4');rgb=np.concatenate(cols)
    np.savez_compressed(OUT/'geometry.npz',xyz=xyz,rgb=rgb)
    record['points']=len(xyz);record['geometry_sha256']=hashlib.sha256((OUT/'geometry.npz').read_bytes()).hexdigest()
    record['status']='INFERENCE_COMPLETE';record['total_seconds']=time.monotonic()-start
    save();print('RESULT',json.dumps(record,default=str),flush=True)
except Exception as e:
    record.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();print(record['traceback'],flush=True);raise
