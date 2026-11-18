"""Use the pinned upstream audit wrapper with the actual late-touch policy."""
import os
from pathlib import Path
from observation_adapter import canonical_observation

root = Path(os.environ['AFCE_EVAL_ROOT'])
source = root/'runtime_private/eval_harness/serve_audited.py'
old = b'from afce_all11.serve_single_finger_pi import main'
new = b'from afce_ss.serve_ss import main'
payload = source.read_bytes()
if payload.count(old) != 1:
    raise ValueError('Pinned audit wrapper changed')
anchor = b'        observation=dict(observation)\n'
if payload.count(anchor) != 1:
    raise ValueError('Pinned audit observation anchor changed')
payload = payload.replace(anchor, b"        observation=canonical_observation(observation, task=IDENTITY['task'])\n")
exec(compile(payload.replace(old, new), str(source), 'exec'),
     {'__name__': '__main__', '__file__': str(source), 'canonical_observation': canonical_observation})
