"""Wait for verified migration and free GPUs, smoke once, then supervise formal evaluation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from supervise_formal import idle_gpus, sha, write


def main():
    root=Path(os.environ['AFCE_EVAL_ROOT']);logs=root/'logs';logs.mkdir(exist_ok=True)
    lock=(logs/'mission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(),True)
    model=root/'models/single_stage_late_touch_touch_seed42'
    status=logs/'MISSION_STATUS.json'
    write(logs/'mission.json',{'pid':os.getpid(),'started_unix':time.time()})
    while True:
        receipt=model/'MIGRATION_VERIFIED.json'
        if not receipt.exists():
            write(status,{'state':'waiting_for_final_model','unix':time.time()});time.sleep(60);continue
        identity=json.loads(receipt.read_text())
        if not identity.get('verified') or identity['optimizer_updates']!=60000:raise ValueError('Wrong model identity')
        for name,row in identity['files'].items():
            if (model/name).stat().st_size!=row['bytes'] or sha(model/name)!=row['sha256']:
                raise ValueError('Published model SHA differs: '+name)
        break
    while True:
        complete=root/'results/real/seed0/complete.json'
        if complete.exists():
            result=json.loads(complete.read_text())
            if result.get('complete') and result.get('verified') and result.get('episodes')==550:
                write(status,{'state':'evaluation_complete','episodes':550,'unix':time.time()});return 0
        smoke=logs/'final_smoke_water_plant/complete.json'
        if smoke.exists() and json.loads(smoke.read_text()).get('passed'):
            # The formal controller continuously adds newly idle cards and waits if none are free.
            cmd=[sys.executable,str(Path(__file__).with_name('supervise_formal.py'))]
        else:
            free=idle_gpus()
            if not free:
                write(status,{'state':'waiting_for_idle_4090','unix':time.time()});time.sleep(60);continue
            cmd=[sys.executable,str(Path(__file__).with_name('smoke_final.py')),'--gpu',str(free[0])]
        write(status,{'state':'running','command':cmd,'unix':time.time()})
        result=subprocess.call(cmd,pass_fds=(lock.fileno(),))
        if result:
            write(status,{'state':'failure_retry_pending','returncode':result,'command':cmd,'unix':time.time()})
            time.sleep(120)


if __name__=='__main__':raise SystemExit(main())
