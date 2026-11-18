"""Atomic small receipts and immutable source identities."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def source_identity(run):
    run = Path(run)
    done = json.loads((run/'complete.json').read_text())
    receipt = json.loads((run/'latest_complete.json').read_text())
    if not done.get('complete') or done['optimizer_updates'] != 60000 or not receipt.get('complete') or receipt['optimizer_updates'] != 60000:
        raise ValueError('Light training requires the complete final 60000-update source model')
    selected = [run/'training_contract.json', run/'decoder_init.npz']
    params = [p for p in (run/'59999/params').rglob('*') if p.is_file()]
    assets = [p for p in (run/'59999/assets').rglob('*') if p.is_file()]
    if not params or not assets:
        raise ValueError('Final checkpoint parameters or normalization assets are absent')
    selected += params+assets
    files = {str(p.relative_to(run)): sha(p) for p in sorted(selected)}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'source_updates': 60000, 'source_fingerprint': identity, 'source_files': files}
