"""Run an unchanged client script (the evaluation client or the warm-up) with the touch patch installed:
run_with_touch.py SCRIPT [its arguments]."""
import runpy
import sys

import client_touch_patch  # noqa: F401

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
