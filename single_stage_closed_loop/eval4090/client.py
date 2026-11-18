"""Synchronous native50Hz rollout with real clock/feedback and durable boundaries."""
import argparse,hashlib,json,os,pickle,random,sys,time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)


def main():
    import imageio,yaml
    import client_touch_patch  # installs the unchanged tactile contact integration
    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import importlib.util
    spec=importlib.util.spec_from_file_location('closed_block_client',Path(__file__).parents[1]/'client.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    BlockClient=module.BlockClient
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((out/'identity.json').read_text())
    np.random.seed(0);random.seed(0)
    env=DexJoCoOpenPIEnv(env_name=cfg['env_name'],camera_mapping=cfg['camera_mapping'],seed=0,
        rand_full=False,randomize_dynamics=False,dual_arm=cfg['robot_type']=='dual_arm',
        prompt=cfg['prompt'],render_mode='rgb_array',pad_state_dim46=True,password=cfg.get('password'))
    rpc=WebsocketClientPolicy(host='127.0.0.1',port=a.port)
    block=BlockClient(rpc);env.start()
    # Capture timestamps at the actual observation/contact sampling site, not RPC arrival.
    original=env._process_obs
    def process(obs):
        result=original(obs);now=float(env._touch_hook.data.time)
        env._feedback_time=now;return result
    env._process_obs=process
    raw=env.env.unwrapped
    assert abs(float(raw.control_dt)-.02)<1e-9
    count=0;boundary=None
    for ep in range(50):
        dest=out/f'episode_{ep:02d}'
        if not (dest/'verified.json').exists():break
        verification=json.loads((dest/'verified.json').read_text());receipt=json.loads((dest/'receipt.json').read_text())
        assert verification['receipt_sha256']==sha(dest/'receipt.json') and receipt['identity']==identity
        assert sha(dest/'progress.pkl')==receipt['progress_sha256']
        for v in receipt['videos']:assert sha(dest/v['file'])==v['sha256']
        count+=1;boundary=pickle.loads((dest/'progress.pkl').read_bytes())
    if boundary:
        np.random.set_state(boundary['numpy_rng']);random.setstate(boundary['python_rng'])
        raw.random_state.set_state(boundary['environment_rng'])
        assert rpc.infer({'_eval_control':'set_rng','rng':boundary['policy_rng']})['restored']
    writers={}
    try:
        for ep in range(count,1 if a.smoke else 50):
            dest=out/f'episode_{ep:02d}';dest.mkdir(exist_ok=True)
            # An interrupted, uncommitted episode is rerun from the last boundary; no duplicate receipt.
            for file in dest.iterdir():
                if file.is_file():file.unlink()
            env.reset();block.reset();steps=0;blocks=[];password=[]
            if cfg['env_name']=='click_mouse':
                for _ in range(30):
                    env.step(np.array([-.44294,1.3729e-6,1.517,-3.14156462,-6.91584035e-5,-.00140317984,
                        0,0,0,0,0,0,0,0,0,0,0,0,.263,0,0,0]))
            writers={cam:imageio.get_writer(dest/(cam+'.mp4'),fps=30) for cam in cfg['camera_mapping'].values()}
            def frame():
                images=env.get_raw_images()
                for cam,w in writers.items():w.append_data(images[cam])
            frame();started=time.monotonic()
            while True:
                obs=env.get_obs();observed=env._feedback_time;actual=float(env._touch_hook.data.time)
                assert abs(actual-observed)<1e-9
                actions=block.infer(obs,observation_time=observed,touch_time=observed)
                offset=block.executed%30;executed=0
                for action in actions:
                    before=float(env._touch_hook.data.time);digits=env.step(action);after=float(env._touch_hook.data.time)
                    assert abs((after-before)-.02)<1e-6
                    steps+=1;executed+=1;frame()
                    if digits:password.append(digits)
                    if env.is_done:break
                if executed==5:block.acknowledge(5)
                blocks.append({'episode_id':block.episode,'offset':offset,'observation_time':observed,
                    'touch_time':observed,'after_time':after,'executed':executed,'finite':True,
                    'touch_sha256':hashlib.sha256(np.asarray(obs['touch']).tobytes()).hexdigest()})
                if env.is_done or (a.smoke and len(blocks)>=3):break
            for w in writers.values():w.close()
            writers={}
            if a.smoke:
                assert len(blocks)>=3 and all(b['executed']==5 for b in blocks)
                write(out/'smoke-complete.json',{'verified':True,'smoke':True,'task':cfg['env_name'],
                    'blocks':blocks,'identity':identity,'steps':steps});return
            assert env.is_done and steps>0
            progress={'next_episode':ep+1,'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'environment_rng':raw.random_state.get_state(),'policy_rng':rpc.infer({'_eval_control':'get_rng'})['rng']}
            (dest/'progress.pkl').write_bytes(pickle.dumps(progress,protocol=5))
            write(dest/'feedback.json',blocks)
            inferred=[json.loads(line) for line in (out/'server-trace.jsonl').read_text().splitlines() if line]
            inferred=[r for r in inferred if r['episode_id']==block.episode]
            assert len(inferred)==len(blocks) and all(r['finite'] for r in inferred)
            write(dest/'inferences.json',inferred)
            videos=[]
            import subprocess
            root=Path(os.environ['AFCE_EVAL_ROOT'])
            for file in sorted(dest.glob('*.mp4')):
                report=subprocess.check_output([sys.executable,str(root/'layout/sim/verify_video.py'),str(file)],text=True)
                v=json.loads(report.strip().splitlines()[-1]);assert v['frames']>1
                videos.append({'file':file.name,'sha256':sha(file),**v})
            assert len(videos)==(3 if cfg['robot_type']=='dual_arm' else 2)
            receipt={'identity':identity,'episode':ep,'task':cfg['env_name'],'seed':0,'smoke':False,
                'success':bool(env.is_success),'steps':steps,'blocks':len(blocks),'password':password,
                'wall_seconds':time.monotonic()-started,'videos':videos,'feedback_sha256':sha(dest/'feedback.json'),
                'inferences_sha256':sha(dest/'inferences.json'),
                'progress_sha256':sha(dest/'progress.pkl')}
            write(dest/'receipt.json',receipt)
            write(dest/'verified.json',{'verified':True,'receipt_sha256':sha(dest/'receipt.json')})
            print('CLOSED_LOOP_EPISODE_VERIFIED',ep,receipt['success'],flush=True)
        write(out/'complete.json',{'complete':True,'verified':True,'episodes':50,'identity':identity})
    finally:
        for w in writers.values():w.close()
        env.close()


if __name__=='__main__':main()
