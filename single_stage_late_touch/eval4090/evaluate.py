"""Portable single-GPU entry point; original 550-episode protocol is retained.

This prepares the published single-stage evaluator in memory. It never starts
training, changes physics, or permits using an incomplete final checkpoint.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ['AFCE_EVAL_ROOT']).resolve()
    if os.environ.get('AFCE_TOUCH_OFF') == '1':
        condition = 'mask0'
    else:
        condition = 'real'
    os.environ.setdefault('EVAL_WORK_ROOT', str(root/'work'/condition))
    os.environ.setdefault('EVAL_RUN_NAME', 'late_touch_'+condition)
    os.environ.setdefault('AFCE_SS_EVAL_TOUCH', '1')
    os.environ.setdefault('EVAL_PAIRS', '1')
    path = root/'repo/single_stage_touch/code/afce_ss/ss_eval.py'
    spec = importlib.util.spec_from_file_location('upstream_single_stage_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N = root/'runtime_private/eval_harness'
    module.SERVE = Path(__file__).with_name('serve_audited.py')
    # Preserve the upstream hash gate and its exact replacements. Only add paths
    # and permit one GPU for the identical formal per-task episode budget.
    module.EDITS = [(a, b.replace(b'(4,8)', b'(1,2,3,4,8)')) for a, b in module.EDITS]
    upstream = module.N/'evaluate_arm.py'
    payload = upstream.read_bytes()
    replacements = [
        (b"N=Path('/opt/zbk/ours-official-replay-matched-20261008/web-resume-20261009')",
         ('N=Path('+repr(str(module.N))+')').encode()),
        (b"R=Path('/opt/zbk/tactile-c01-baseline-20261005');W=Path('/opt/zbk/v2m_eval_seed0_20261002')",
         ('R=Path('+repr(str(root/'layout/policy'))+');W=Path('+repr(str(root/'layout/sim'))+')').encode()),
        (b"PY=Path('/opt/zbk/afce-pi/bin/python');SIM=W/'simenv/bin/python'",
         ('PY=Path('+repr(str(root/'venv/bin/python'))+");SIM=W/'simenv/bin/python'").encode()),
    ]
    for old, _ in replacements:
        if payload.count(old) != 1:
            raise ValueError('Upstream path anchor changed')
    module.EDITS = replacements+module.EDITS
    portable = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('evaluate.py', 'serve_audited.py', 'observation_adapter.py',
                             'run_formal.py', 'supervise_formal.py')}
    module.EDITS.append((b"'single_stage_code':_CODE_SHA}",
                         b"'single_stage_code':_CODE_SHA,'portable_eval_code':"+repr(portable).encode()+b"}"))
    if '--check-patches' in sys.argv:
        if hashlib.sha256(payload).hexdigest() != module.SOURCE_SHA:
            raise ValueError('Pinned evaluation source hash differs')
        edited = payload
        for old, new in module.EDITS+module.TOUCH_EDITS:
            if edited.count(old) != 1:
                raise ValueError('Portable evaluation anchor differs: '+repr(old))
            edited = edited.replace(old, new)
        compile(edited, str(upstream), 'exec')
        print('EVALUATION_PATCHES_VERIFIED', flush=True)
        return
    print('PORTABLE_EVAL_PATCHES', json.dumps({
        'source_sha256': hashlib.sha256(payload).hexdigest(),
        'scope': 'paths and GPU count; rand_obj, H30, 10 denoising steps, replan .8, 50 episodes/task unchanged',
        'condition': condition,
    }), flush=True)
    module.main()


if __name__ == '__main__':
    main()
