"""One real reset-observation inference after verified final-model migration.

No episode or success-rate measurement. Use an idle GPU; do not preempt jobs.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--task', choices=['water_plant', 'bimanual_assembly'], default='water_plant')
    args = parser.parse_args()
    root = Path(os.environ['AFCE_EVAL_ROOT'])
    run = root/'models/single_stage_late_touch_touch_seed42'
    receipt = json.loads((run/'MIGRATION_VERIFIED.json').read_text())
    if not receipt.get('verified') or receipt['optimizer_updates'] != 60000:
        raise ValueError('Only the verified final checkpoint may be loaded')
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    memory = {int(x.split(',')[0]): int(x.split(',')[1]) for x in usage.splitlines()}
    if memory[args.gpu] > 64:
        print('SMOKE_DEFERRED_GPU_BUSY', args.gpu, memory[args.gpu], flush=True)
        return 10
    folder = root/'logs'/('final_smoke_'+args.task)
    folder.mkdir(exist_ok=True)
    port = 26700+args.gpu
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    identity = {'arm': 'ours', 'task': args.task, 'optimizer_updates': 60000, 'rng_offset': 0,
                'kind': 'reset_observation_inference_only', 'performance_claimed': False}
    (folder/'identity.json').write_text(json.dumps(identity))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), EVAL_IDENTITY=str(folder/'identity.json'),
               EVAL_SERVER_TRACE=str(folder/'server-trace.jsonl'))
    env.pop('JAX_PLATFORMS', None)
    cmd = [str(root/'venv/bin/python'), '-u', str(Path(__file__).with_name('serve_audited.py')),
           '--checkpoint', str(run/'59999'), '--ss-run', str(run), '--data', str(root/'rebuilt_native_v1/datasets'),
           '--base', str(run/'59999/params'), '--output', str(root/'work/final_smoke'),
           '--name', 'final_smoke_'+args.task, '--task', args.task, '--port', str(port)]
    with (folder/'server.log').open('w') as log:
        server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            entered = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy exited {server.returncode}; see {folder}/server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic()-entered > 900:
                    raise TimeoutError('Policy initialization timeout')
                time.sleep(2)
            client_env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str(args.gpu))
            client_env.pop('JAX_PLATFORMS', None)
            client_env.pop('CUDA_VISIBLE_DEVICES', None)
            code = root/'repo/single_stage_touch/code'
            client_env['PYTHONPATH'] = str(code/'sim_touch')+':'+client_env['PYTHONPATH']
            with (folder/'warmup.log').open('w') as log:
                subprocess.run([str(root/'venv/bin/python'), '-u', str(code/'sim_touch/run_with_touch.py'),
                    str(root/'runtime_private/eval_harness/warmup.py'), '--config',
                    str(root/'sim_source/configs/rand_obj'/f'{args.task}.yaml'), '--port', str(port)],
                    env=client_env, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
            result = {**identity, 'passed': True, 'finished_unix': time.time()}
            (folder/'complete.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
