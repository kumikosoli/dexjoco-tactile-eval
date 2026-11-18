"""Explicit session/acknowledgement policy server; legacy H30 clients are rejected."""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'correction-run', 'data', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, default=26803)
    a = p.parse_args()
    from afce_cl.serving import light_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = light_policy(a.run, a.correction_run, a.data, a.output, task=a.task)
    WebsocketPolicyServer(policy=policy, host='127.0.0.1', port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
