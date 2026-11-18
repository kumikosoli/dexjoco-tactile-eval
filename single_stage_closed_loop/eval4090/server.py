"""Isolated audited five-action policy server; no changes to the static evaluator."""
import argparse,json,os,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from afce_cl.serving import light_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from observation_adapter import canonical_observation
    import jax
    import jax.numpy as jnp
    root=Path(os.environ['AFCE_EVAL_ROOT']);light=root/'closed_loop_eval'
    policy=light_policy(root/'models/single_stage_late_touch_touch_seed42',light/'light_seed42',
        root/'rebuilt_native_v1/datasets',a.output,task=a.task)
    class Audited:
        metadata=policy.metadata
        def reset(self):policy.reset()
        def infer(self,observation,**kwargs):
            control=observation.get('_eval_control')
            if control=='get_rng':return {'rng':np.asarray(jax.random.key_data(policy.latent._rng))}
            if control=='set_rng':
                key=np.asarray(observation['rng']);assert key.shape==(2,) and key.dtype==np.uint32
                policy.latent._rng=jax.random.wrap_key_data(jnp.asarray(key));return {'restored':True}
            obs=canonical_observation(observation,task=a.task)
            begin=time.monotonic();result=policy.infer(obs)
            actions=np.asarray(result['actions']);assert actions.shape==(5,44 if a.task.startswith('bimanual_') else 22)
            assert np.isfinite(actions).all()
            row={'utc':time.time(),'seconds':time.monotonic()-begin,'finite':True,
                 'episode_id':obs['closed_loop']['episode_id'],**result['closed_loop']}
            with (a.output/'server-trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return result
    a.output.mkdir(parents=True,exist_ok=True)
    WebsocketPolicyServer(policy=Audited(),host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__':main()
