"""Use newly idle 4090s for disjoint whole-task shards; finish only after 550 verified episodes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASKS=['bimanual_assembly','bimanual_hanoi','pinch_tongs','fold_glasses','bimanual_microwave_cook',
       'hammer_nail','bimanual_unlock_ipad','pick_bucket','bimanual_photograph','click_mouse','water_plant']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def idle_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
    idle=[]
    for line in raw.splitlines():
        index,uuid,name,memory,usage=[v.strip() for v in line.split(',')]
        if '4090' in name and uuid not in busy and int(memory)<=64 and int(usage)<=5:idle.append(int(index))
    return idle


def verified_task(root,task):
    shard=root/'shards'/task
    complete=shard/'complete.json'
    if not complete.exists():return None
    result=json.loads(complete.read_text())
    if not result.get('complete') or result['episodes']!=50:raise ValueError('Wrong task completion budget')
    receipts=[]
    for ep in range(50):
        dest=shard/task/f'episode_{ep:02d}'
        verified=json.loads((dest/'verified.json').read_text())
        if not verified.get('verified') or sha(dest/'receipt.json')!=verified['receipt_sha256']:
            raise ValueError('Invalid episode receipt')
        row=json.loads((dest/'receipt.json').read_text())
        if (row['task']!=task or row['episode']!=ep or row['seed']!=0 or row['smoke']
            or row['optimizer_updates']!=60000 or row['level']!='rand_obj' or row['denoising_steps']!=10
            or row['replan_ratio']!=.8 or row['episodes_per_task']!=50 or row['run']!='late_touch_real'):
            raise ValueError('Episode identity/protocol differs')
        if sha(dest/'progress.pkl')!=row['progress_sha256']:raise ValueError('Progress SHA differs')
        for video in row['videos']:
            if sha(dest/video['file'])!=video['sha256']:raise ValueError('Video SHA differs')
        receipts.append(row)
    if len({x['checkpoint_sha256'] for x in receipts})!=1:raise ValueError('Mixed checkpoints within task')
    return {'task':task,'episodes':50,'successes':sum(r['success'] for r in receipts),
        'checkpoint_sha256':receipts[0]['checkpoint_sha256'],
        'receipts':{f'{task}/{r["episode"]}':sha(shard/task/f'episode_{r["episode"]:02d}'/'receipt.json') for r in receipts}}


def main():
    base=Path(os.environ['AFCE_EVAL_ROOT']);root=base/'results/real/seed0';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Propagate this lock to task launchers so a restarted controller cannot duplicate surviving children.
    os.set_inheritable(lock.fileno(),True)
    write(root/'supervisor.json',{'pid':os.getpid(),'started_unix':time.time(),'protocol':'real/seed0/11x50'})
    running={};cooldown={};done={}
    while True:
        for task in TASKS:
            if task not in done and task not in running:
                result=verified_task(root,task)
                if result:done[task]=result
        for task,(proc,gpu,log) in list(running.items()):
            if proc.poll() is None:continue
            log.close();del running[task]
            if proc.returncode:cooldown[task]=time.time()+120
            else:
                result=verified_task(root,task)
                if result is None:raise RuntimeError('Launcher exited without complete verified shard: '+task)
                done[task]=result
        if len(done)==len(TASKS):
            # Final pass checks every unique receipt and referenced progress/video byte.
            final=[verified_task(root,task) for task in TASKS]
            if len({r['checkpoint_sha256'] for r in final})!=1:raise ValueError('Mixed model checkpoints')
            episodes={k:v for r in final for k,v in r['receipts'].items()}
            if len(episodes)!=550:raise ValueError('Need exactly 550 unique task/episode pairs')
            write(root/'complete.json',{'complete':True,'verified':True,'episodes':550,
                'successes':sum(r['successes'] for r in final),'tasks':final,'condition':'real',
                'evaluation_seed':0,'optimizer_updates':60000,'completed_unix':time.time()})
            (root/'REPORT.md').write_text('# Real tactile final evaluation\n\n'+
                f'Verified episodes: 550; successes: {sum(r["successes"] for r in final)}/550.\n\n'+
                '\n'.join(f'- {r["task"]}: {r["successes"]}/50' for r in final)+'\n')
            return 0
        active={gpu for _,gpu,_ in running.values()}
        pending=[task for task in TASKS if task not in done and task not in running and time.time()>=cooldown.get(task,0)]
        for gpu in idle_gpus():
            if gpu in active or not pending:continue
            task=pending.pop(0);log=(root/f'{task}.launcher.log').open('a',buffering=1)
            cmd=[sys.executable,str(Path(__file__).with_name('run_formal.py')),'--gpu',str(gpu),
                 '--condition','real','--seed','0','--task',task,'--persist',str(root/'shards'/task)]
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
            running[task]=(proc,gpu,log)
        write(root/'STATUS.json',{'complete_tasks':list(done),'completed_episodes':50*len(done),
            'running':{task:{'pid':p.pid,'gpu':gpu} for task,(p,gpu,_) in running.items()},
            'waiting_tasks':[task for task in TASKS if task not in done and task not in running],
            'updated_unix':time.time()})
        time.sleep(30)


if __name__=='__main__':raise SystemExit(main())
