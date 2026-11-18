"""Wait for old550, validate feedback smoke, then schedule disjoint light550 shards."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path

TASKS=['bimanual_assembly','bimanual_hanoi','pinch_tongs','fold_glasses','bimanual_microwave_cook',
       'hammer_nail','bimanual_unlock_ipad','pick_bucket','bimanual_photograph','click_mouse','water_plant']
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    result=[]
    for line in raw.splitlines():
        idx,uuid,name,mem,use=map(str.strip,line.split(','))
        if '4090' in name and uuid not in busy and int(mem)<=64 and int(use)<=5:result.append(int(idx))
    return result
def identity(root,task):
    light=root/'closed_loop_eval';run=root/'models/single_stage_late_touch_touch_seed42'
    from afce_cl.io import source_identity
    source=source_identity(run)
    contract=json.loads((light/'light_seed42/contract.json').read_text())
    assert contract['source']['source_fingerprint']==source['source_fingerprint']
    receipt=json.loads((light/'LIGHT_MIGRATION_VERIFIED.json').read_text());assert receipt['verified'] and receipt['step']==10000
    for n,v in receipt['files'].items():assert sha(light/n)==v['sha256']
    code={str(p.relative_to(light)):sha(p) for base in ('repo','runtime_private_v2')
          for p in sorted((light/base).rglob('*.py')) if '__pycache__' not in str(p)}
    off=root/'layout/sim/dexjoco-8d23b0fab23b17a58c4b55f3942e17013aaf8267'
    simulator={str(p.relative_to(off)):sha(p) for p in sorted((off/'dexjoco').rglob('*.py'))}
    simulator['task_config']=sha(off/'configs/rand_obj'/f'{task}.yaml')
    return {'method':'afce_complete_E_block_feedback_v1','task':task,'seed':0,'training_seed':42,
        'source_updates':60000,'correction_updates':10000,'source_fingerprint':source['source_fingerprint'],
        'correction_sha256':sha(light/'light_seed42/latest.pkl'),'code':code,'simulator':simulator,'level':'rand_obj',
        'denoising_steps':10,'horizon':30,'feedback_block':5,'control_hz':50,'episodes_per_task':50,
        'timing_difference':'synchronous feedback5/E30; static baseline async replan0.8',
        'official_commit':'8d23b0fab23b17a58c4b55f3942e17013aaf8267'}
def verify(out,ident,full=False):
    receipts=[]
    for ep in range(50):
        d=out/f'episode_{ep:02d}'
        if not (d/'verified.json').exists():break
        v=json.loads((d/'verified.json').read_text());r=json.loads((d/'receipt.json').read_text())
        assert v['verified'] and v['receipt_sha256']==sha(d/'receipt.json')
        assert r['identity']==ident and r['episode']==ep and r['task']==ident['task'] and r['seed']==0 and not r['smoke']
        assert sha(d/'progress.pkl')==r['progress_sha256'] and sha(d/'feedback.json')==r['feedback_sha256']
        rows=json.loads((d/'feedback.json').read_text());assert rows and all(x['finite'] for x in rows)
        assert sha(d/'inferences.json')==r['inferences_sha256']
        inference=json.loads((d/'inferences.json').read_text())
        assert len(inference)==len(rows) and all(x['finite'] for x in inference)
        for i,b in enumerate(rows):
            assert b['offset']==(i*5)%30 and b['touch_time']==b['observation_time']
            assert abs(b['after_time']-b['observation_time']-b['executed']*.02)<1e-5
            assert b['executed']==5 or (i==len(rows)-1 and b['executed'] in range(1,5))
            assert inference[i]['start_offset']==b['offset'] and inference[i]['touch_time']==b['touch_time']
        assert len(r['videos'])==(3 if ident['task'].startswith('bimanual_') else 2)
        for video in r['videos']:assert sha(d/video['file'])==video['sha256']
        receipts.append(r)
    if full:assert len(receipts)==50 and json.loads((out/'complete.json').read_text())['verified']
    return receipts
def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def task_run(root,task,gpu,smoke):
    light=root/'closed_loop_eval';out=light/('smoke' if smoke else 'results')/task;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'task.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ident=identity(root,task)
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==ident
    else:write(out/'identity.json',ident)
    assert gpu in idle(),'GPU became busy; wait'
    port=27800+gpu
    penv={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}
    cenv={**os.environ,'MUJOCO_EGL_DEVICE_ID':str(gpu)};cenv.pop('CUDA_VISIBLE_DEVICES',None)
    sim=root/'layout/sim/simenv/bin/python';off=root/'layout/sim/dexjoco-8d23b0fab23b17a58c4b55f3942e17013aaf8267'
    server=None
    with (out/'server.log').open('a') as log:
        try:
            server=subprocess.Popen([sys.executable,'-u',str(HERE/'server.py'),'--task',task,'--port',str(port),'--output',str(out)],env=penv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError('policy startup failed: '+str(out))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=1) as response:
                        if response.status==200:break
                except OSError:pass
                if time.monotonic()-started>900:raise TimeoutError('server loading')
                time.sleep(2)
            with (out/'client.log').open('a') as clientlog:
                subprocess.run([str(sim),'-u',str(HERE/'client.py'),'--config',str(off/'configs/rand_obj'/f'{task}.yaml'),
                    '--port',str(port),'--output',str(out)]+(['--smoke'] if smoke else []),env=cenv,stdout=clientlog,stderr=subprocess.STDOUT,check=True)
            if smoke:
                report=json.loads((out/'smoke-complete.json').read_text());assert report['verified'] and len(report['blocks'])>=3
            else:verify(out,ident,full=True)
        finally:stop(server)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS);parser.add_argument('--gpu',type=int)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    root=Path(os.environ['AFCE_EVAL_ROOT']);light=root/'closed_loop_eval'
    if a.task:return task_run(root,a.task,a.gpu,a.smoke)
    lock=(light/'mission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(),True)
    running={};cooldown={}
    while True:
        prior=root/'results/real/seed0/complete.json'
        if not prior.exists() or not json.loads(prior.read_text()).get('verified') or json.loads(prior.read_text()).get('episodes')!=550:
            write(light/'STATUS.json',{'stage':'waiting_for_static550','pid':os.getpid()});time.sleep(30);continue
        for task,(p,g,log) in list(running.items()):
            if p.poll() is not None:
                log.close();del running[task]
                if p.returncode:cooldown[task]=time.time()+180
        smoke_done=all((light/'smoke'/t/'smoke-complete.json').exists() for t in ('water_plant','bimanual_assembly'))
        done=[t for t in TASKS if (light/'results'/t/'complete.json').exists()]
        if len(done)==11:
            results=[]
            for task in TASKS:
                rs=verify(light/'results'/task,identity(root,task),full=True)
                results.append({'task':task,'episodes':50,'successes':sum(r['success'] for r in rs)})
            write(light/'complete.json',{'complete':True,'verified':True,'episodes':550,'tasks':results,'successes':sum(r['successes'] for r in results)})
            (light/'REPORT.md').write_text('# Lightweight closed-loop evaluation\n\n'+str(results)+'\n\nSynchronous feedback5/E30 differs from static async replan0.8.\n');return
        pending=([t for t in ('water_plant','bimanual_assembly') if not (light/'smoke'/t/'smoke-complete.json').exists()] if not smoke_done else [t for t in TASKS if t not in done])
        pending=[t for t in pending if t not in running and time.time()>=cooldown.get(t,0)]
        for gpu in idle():
            if not pending or gpu in {g for _,g,_ in running.values()}:continue
            task=pending.pop(0);log=(light/(task+'.launcher.log')).open('a')
            cmd=[sys.executable,'-u',str(__file__),'--task',task,'--gpu',str(gpu)]+([] if smoke_done else ['--smoke'])
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
            running[task]=(p,gpu,log)
        write(light/'STATUS.json',{'stage':'formal' if smoke_done else 'smoke','pid':os.getpid(),'done_tasks':done,
            'running':{t:{'pid':p.pid,'gpu':g} for t,(p,g,l) in running.items()},'unix':time.time()})
        time.sleep(30)


if __name__=='__main__':main()
