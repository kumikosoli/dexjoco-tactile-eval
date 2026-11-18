"""Reset and hold two native tasks; verifies assets/rendering/current touch only."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

root = Path(os.environ['AFCE_EVAL_ROOT'])
sys.path.insert(0, str(root/'repo/single_stage_touch/code/sim_touch'))
import client_touch_patch  # installs the same read-only tactile hook as evaluation
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

for task in ['water_plant', 'bimanual_assembly']:
    cfg = yaml.safe_load((root/'sim_source/configs/rand_obj'/f'{task}.yaml').read_text())
    env = DexJoCoOpenPIEnv(env_name=cfg['env_name'], camera_mapping=cfg['camera_mapping'], seed=0,
        rand_full=False, randomize_dynamics=False, dual_arm=cfg['robot_type'] == 'dual_arm',
        prompt=cfg['prompt'], render_mode='rgb_array', pad_state_dim46=True, password=cfg.get('password'))
    env.start()
    try:
        env.reset()
        env.stay()
        obs = env.get_obs()
        assert obs['touch'].shape == (8, 166) and obs['touch_mask'].shape == (8,)
        assert np.isfinite(obs['touch']).all() and obs['touch_mask'].sum() == (8 if task.startswith('bimanual') else 4)
        assert env._touch_hook.substeps == env._touch_hook.expected
        images = env.get_raw_images()
        assert images and all(np.asarray(v).ndim == 3 and np.asarray(v).std() > 0 for v in images.values())
        print(json.dumps({'task': task, 'state': obs['state'].shape, 'touch': obs['touch'].shape,
            'present_fingers': int(obs['touch_mask'].sum()), 'physics_substeps': env._touch_hook.substeps,
            'cameras': list(images), 'passed': True}), flush=True)
    finally:
        env.close()
