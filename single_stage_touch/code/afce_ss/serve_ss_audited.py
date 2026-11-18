"""The published serve_audited.py (request audit/trace wrapper), serving the single-stage policy."""
from pathlib import Path
import hashlib

SOURCE = Path('/opt/zbk/ours-official-replay-matched-20261008/web-resume-20261009/serve_audited.py')
OLD = b'from afce_all11.serve_single_finger_pi import main'
NEW = b'from afce_ss.serve_ss import main'
source = SOURCE.read_bytes()
assert source.count(OLD) == 1, 'serve_audited.py changed'
exec(compile(source.replace(OLD, NEW), str(SOURCE), 'exec'), {'__name__': '__main__', '__file__': str(SOURCE)})
