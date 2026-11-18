"""Load the existing final model once for prediction caching or block serving."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys


def load_frozen(run, data, output):
    run, data, output = map(Path, (run, data, output))
    runtime = Path(os.environ['AFCE_OPENPI_ROOT'])
    sys.path.insert(0, str(runtime/'scripts'))
    import dexjoco_11_task_balanced as baseline
    from serve_dexjoco_11_task_checkpoint import _load_jax_model
    from afce_ss.ss_late_touch import SingleStageLateTouchPiConfig
    from afce_ss.ss_model import SingleStageDataConfig
    from openpi.policies.policy import Policy
    from openpi.training import checkpoints
    from openpi import transforms
    contract = json.loads((run/'training_contract.json').read_text())
    trained = contract['model']
    if not trained.get('late_touch') or trained.get('closed_block_size') is not None:
        raise ValueError('Expected the first complete-E late-touch baseline, not a joint closed-loop run')
    cfg, _, _ = baseline.build_config(baseline.Args(operation='inspect', data_root=data,
        init_params_path=run/'59999/params', exp_name='frozen_closed_cache',
        checkpoint_base_dir=output/'checkpoints', assets_base_dir=output/'assets'))
    cfg = dataclasses.replace(cfg, name='afce_all11_official', data=dataclasses.replace(cfg.data, balance='proportional'))
    kwargs = {k: trained[k] for k in ('touch_scale', 'late_touch_stats_sha256', 'late_touch_width',
                                    'late_touch_limit', 'late_touch_dropout', 'late_touch_mask0')}
    kwargs['touch_scale'] = tuple(kwargs['touch_scale'])
    init = run/'decoder_init.npz'
    digest = hashlib.sha256(init.read_bytes()).hexdigest()
    if digest != trained['decoder_init_sha256']:
        raise ValueError('Source decoder init differs')
    model = SingleStageLateTouchPiConfig(**dataclasses.asdict(cfg.model), **kwargs,
        decoder_init_path=str(init), decoder_init_sha256=digest, decoder_warmup_steps=0,
        alignment_target_mode='ground_truth', alignment_start=trained['alignment_start'],
        alignment_ramp_steps=trained['alignment_ramp_steps'], joint_decoder_lr_scale=trained['joint_decoder_lr_scale'])
    data_cfg = SingleStageDataConfig(**{f.name: getattr(cfg.data, f.name) for f in dataclasses.fields(cfg.data)},
        afce_cache_root=run/'single_stage_cache', touch=True)
    cfg = dataclasses.replace(cfg, model=model, data=data_cfg)
    loaded = _load_jax_model(cfg, run/'59999')
    pipeline = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = checkpoints.load_norm_stats(run/'59999/assets', pipeline.asset_id)
    policy = Policy(loaded, transforms=[transforms.InjectDefaultPrompt(None), *pipeline.data_transforms.inputs,
        transforms.Normalize(norm, use_quantiles=pipeline.use_quantile_norm), *pipeline.model_transforms.inputs],
        output_transforms=[], metadata={'source': 'frozen_final_late_touch', 'performance_claimed': False})
    return policy, loaded
