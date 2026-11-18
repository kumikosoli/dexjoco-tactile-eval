"""Published evaluate_arm.py for a single-stage AFCE run: same official client, simulator, rand_obj level,
10 denoising steps, replan ratio 0.8, episode verification and durable copies as the two-stage runs.
Changed in memory only:
  * the frozen-E-cache checks are replaced by the single-stage run identity (--query = run root);
  * the policy server is serve_ss_audited.py (single-stage config; same audit wrapper);
  * evaluation seed from EVAL_SEED (default 0); 4 or 8 GPUs; PAIRS policy+simulator pairs per card;
  * work root / run name from EVAL_WORK_ROOT / EVAL_RUN_NAME;
  * AFCE_SS_EVAL_TOUCH=1 (a run trained with --touch): as scheme B's evaluation (tacab/exp_eval_b.py), the server
    imports scheme B's patched code copy (AFCE_CODE_ROOT) and the warm-up and client run through
    sim_touch/run_with_touch.py, which adds the current fingertip touch to each policy observation;
    AFCE_TOUCH_OFF=1 then masks every tactile token (the no-touch control of the same checkpoint).
"""
from pathlib import Path
import hashlib, json, os, sys, threading, time

N = Path('/opt/zbk/ours-official-replay-matched-20261008/web-resume-20261009')
SOURCE_SHA = '2d3b852b1983006a28ead3c1615521713c1266572f3c8a9900f19fcf30ebe683'
CODE = Path(__file__).resolve().parents[1]
SERVE = CODE/'afce_ss'/'serve_ss_audited.py'
PAIRS = int(os.environ.get('EVAL_PAIRS', '2'))
SEED = int(os.environ.get('EVAL_SEED', '0'))
WORK = Path(os.environ['EVAL_WORK_ROOT'])
NAME = os.environ['EVAL_RUN_NAME']
OLD_ASSETS = b"""  manifest=json.loads((query_root/'effect_cache/manifest.json').read_text())
  assert manifest['complete'] and (manifest['codec_step']==12 if args.mode=='debug' else (0<manifest['codec_step']<=30000 and manifest['training_completed_updates']==30000))
  from afce_all11.ablation_assets import validate_policy_assets,validate_data
  validate_policy_assets(checkpoint,query_root/'effect_cache',query_root/'joint_decoder_init.npz','ASV')
  validate_data(query_root/'effect_cache',data_root)
"""
NEW_ASSETS = b"""  manifest=_ss_manifest(checkpoint,query_root)
"""
EDITS = [
    (OLD_ASSETS, NEW_ASSETS),
    (b"(len(gpus)==4 and episodes==50)", b"(len(gpus) in (4,8) and episodes==50)"),
    (b"'evaluation_seed':0,", b"'evaluation_seed':_SEED,"),
    (b"row['seed']==0 and", b"row['seed']==_SEED and"),
    (b"'--seed','0',", b"'--seed',str(_SEED),"),
    (b"port=25700+gpu;", b"port=_pair_port(gpu);"),
    (b"ThreadPoolExecutor(max_workers=len(gpus)) as pool", b"ThreadPoolExecutor(max_workers=len(gpus)*_PAIRS) as pool"),
    (b"pool.map(worker,gpus)", b"pool.map(worker,gpus*_PAIRS)"),
    (b"attempt=N/'eval-work'/args.dataset_arm/args.mode/task/", b"attempt=_WORK/'eval-work'/f'seed{_SEED}'/args.mode/task/"),
    (b"'--output',str(N/'policy-config'/args.dataset_arm)", b"'--output',str(_WORK/'policy-config')"),
    (b"'--name','matched_'+args.dataset_arm+'_'+args.mode", b"'--name',_NAME+'_'+args.mode"),
    (b"'episodes_per_task':episodes,'mode':args.mode}", b"'episodes_per_task':episodes,'mode':args.mode,'run':_NAME,'single_stage_code':_CODE_SHA}"),
    (b"server_args=[str(PY),'-u',str(N/'serve_audited.py'),", b"server_args=[str(PY),'-u',str(_SERVE),"),
    (b"if args.arm=='ours':server_args+=['--condition','ASV','--effect-cache',str(query_root/'effect_cache'),'--joint-decoder-init',str(query_root/'joint_decoder_init.npz'),'--single-finger']",
     b"if args.arm=='ours':server_args+=['--ss-run',str(query_root)]"),
    (b"'PYTHONPATH':':'.join(map(str,[W/'eval_extra',R/'code',", b"'PYTHONPATH':':'.join(map(str,[_CODE,W/'eval_extra',R/'code',"),
]
TOUCH = os.environ.get('AFCE_SS_EVAL_TOUCH') == '1'
TOUCH_EDITS = [
    (b"'XLA_PYTHON_CLIENT_ALLOCATOR':'platform','AFCE_OPENPI_ROOT':str(R/'code/openpi'),",
     b"'XLA_PYTHON_CLIENT_ALLOCATOR':'platform','AFCE_OPENPI_ROOT':str(_B/'openpi'),"),
    # (after EDITS, which already put _CODE in front of the server's path)
    (b"'PYTHONPATH':':'.join(map(str,[_CODE,W/'eval_extra',R/'code',R/'code/openpi/src',R/'code/openpi/scripts',R/'code/openpi/packages/openpi-client/src']))}",
     b"'PYTHONPATH':':'.join(map(str,[_CODE,W/'eval_extra',_B,_B/'openpi/src',_B/'openpi/scripts',_B/'openpi/packages/openpi-client/src']))}"),
    (b"'PYTHONPATH':':'.join(map(str,[OFF/'dexjoco',R/'code/openpi/packages/openpi-client/src']))}",
     b"'PYTHONPATH':':'.join(map(str,[_CODE/'sim_touch',_CODE,OFF/'dexjoco',R/'code/openpi/packages/openpi-client/src']))}"),
    (b"subprocess.run([str(SIM),'-u',str(W/'warmup.py'),'--config'",
     b"subprocess.run([str(SIM),'-u',str(_CODE/'sim_touch/run_with_touch.py'),str(W/'warmup.py'),'--config'"),
    (b"client=subprocess.Popen([str(SIM),'-u',str(N/'client_resumable.py'),'--config'",
     b"client=subprocess.Popen([str(SIM),'-u',str(_CODE/'sim_touch/run_with_touch.py'),str(N/'client_resumable.py'),'--config'"),
]
local = threading.local()
lock = threading.Lock()
taken = {}


STAGGER = float(os.environ.get('EVAL_STAGGER_SECONDS', 30))
started = []


def pair_port(gpu):
    # Staggered first starts: simultaneous first-inference JIT of ~11 servers takes ~46 s, longer than the
    # warm-up client's 40 s websocket keepalive (2026-10-15 r2, 2026-10-12 scheme B). A lone start takes ~13 s.
    if not hasattr(local, 'pair'):
        with lock:
            local.pair = taken.get(gpu, 0)
            taken[gpu] = local.pair+1
            local.order = len(started)
            started.append(gpu)
        assert local.pair < PAIRS+1
        print('EVAL_STAGGER', json.dumps({'order': local.order, 'gpu': gpu, 'wait_seconds': STAGGER*local.order}), flush=True)
        time.sleep(STAGGER*local.order)
    return 25700+gpu+8*local.pair


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ss_manifest(checkpoint, run_root):
    run_root = Path(run_root)
    updates = int(checkpoint.name)+1
    encoder = run_root/f'encoder-{updates}.pt'
    contract = json.loads((run_root/'training_contract.json').read_text())
    if contract['schema'] != 'afce_single_stage_v1' or not encoder.is_file() or checkpoint.parent != run_root:
        raise ValueError('Checkpoint is not a complete single-stage run checkpoint')
    return {'codec_step': None, 'training_completed_updates': updates, 'checkpoint_sha256': sha(encoder),
            'selection_sha256': None, 'single_stage_contract_sha256': sha(run_root/'training_contract.json')}


def main():
    source = (N/'evaluate_arm.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    for old, new in EDITS+(TOUCH_EDITS if TOUCH else []):
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    code_sha = {p.name: sha(p) for p in sorted((CODE/'afce_ss').glob('*.py'))}
    print('SINGLE_STAGE_EVAL_LAYOUT', json.dumps({'run': NAME, 'seed': SEED, 'pairs_per_gpu': PAIRS, 'touch': TOUCH,
          'touch_off': os.environ.get('AFCE_TOUCH_OFF') == '1', 'stagger_seconds': STAGGER,
          'evaluate_arm_sha256': SOURCE_SHA, 'code': code_sha}), flush=True)
    sys.argv[0] = str(N/'evaluate_arm.py')
    exec(compile(source, str(N/'evaluate_arm.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(N/'evaluate_arm.py'), '_pair_port': pair_port, '_PAIRS': PAIRS,
          '_WORK': WORK, '_NAME': NAME, '_SEED': SEED, '_SERVE': SERVE, '_CODE': CODE, '_CODE_SHA': code_sha,
          '_ss_manifest': ss_manifest, '_B': Path(os.environ.get('AFCE_CODE_ROOT', '/nonexistent'))})


if __name__ == '__main__':
    main()
