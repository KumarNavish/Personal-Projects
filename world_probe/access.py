import requests,json,time
from pathlib import Path
from gradio_client import Client
out=Path('world_output');out.mkdir(exist_ok=True)
for space in ['prithivMLmods/HY-World-2.0-Demo','Bhamburger/hyworld-pano-wb']:
    d={'space':space,'inference_run':False}
    for filename in ['app.py','README.md']:
        try:
            r=requests.get('https://huggingface.co/spaces/'+space+'/raw/main/'+filename,timeout=30)
            d[filename]={'status':r.status_code,'source':r.text[:50000]}
        except Exception as e:d[filename]={'error':repr(e)}
    try:d['api']=Client(space,verbose=False,httpx_kwargs={'timeout':40}).view_api(return_format='dict',print_info=False)
    except Exception as e:d['api_error']=repr(e)
    name=space.split('/')[-1];(out/(name+'.json')).write_text(json.dumps(d))
    print('RESULT_BEGIN '+name+'\n'+json.dumps(d)+'\nRESULT_END',flush=True)
