"""Authorized formal evaluation with idle GPU selection and optional task shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--condition', choices=['real', 'mask0'], default='real')
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--gpus', help='comma-separated idle GPUs; overrides --gpu')
parser.add_argument('--task', action='append')
parser.add_argument('--persist', type=Path)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
root = Path(os.environ['AFCE_EVAL_ROOT'])
run = root/'models/single_stage_late_touch_touch_seed42'
receipt = json.loads((run/'MIGRATION_VERIFIED.json').read_text())
assert receipt['verified'] and receipt['optimizer_updates'] == 60000
usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                 '--format=csv,noheader,nounits'], text=True)
memory = {int(x.split(',')[0]): int(x.split(',')[1]) for x in usage.splitlines()}
gpus = [int(v) for v in args.gpus.split(',')] if args.gpus else [args.gpu]
if not gpus or len(gpus) != len(set(gpus)):
    raise ValueError('Distinct GPU ids required')
if any(memory[g] > 64 for g in gpus):
    raise RuntimeError('Requested GPU is occupied; no preemption')
env = dict(os.environ, EVAL_SEED=str(args.seed), EVAL_PAIRS='1',
           EVAL_RUN_NAME='late_touch_'+args.condition,
           EVAL_WORK_ROOT=str(root/'work'/args.condition))
env.pop('JAX_PLATFORMS', None)
if args.condition == 'mask0':
    env['AFCE_TOUCH_OFF'] = '1'
else:
    env.pop('AFCE_TOUCH_OFF', None)
cmd = [sys.executable, str(Path(__file__).with_name('evaluate.py')), '--mode', 'formal', '--arm', 'ours',
       '--dataset-arm', 'replay', '--checkpoint', str(run/'59999'), '--query', str(run),
       '--data', str(root/'rebuilt_native_v1/datasets'), '--gpus', ','.join(map(str, gpus)),
       '--persist', str(args.persist or root/'results'/args.condition/f'seed{args.seed}')]
for task in args.task or []:
    cmd += ['--task', task]
raise SystemExit(subprocess.call(cmd, env=env))
