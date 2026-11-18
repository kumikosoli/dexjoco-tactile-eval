"""Serve a single-stage AFCE policy. Same input processing, sampling and decoded-action path as
afce_all11.serve_single_finger_pi (--single-finger branch); only the model/data config is built from the
single-stage run (its decoder_init.npz, effect mean 0 / std 1) instead of a frozen E cache. A run trained with
--touch is served with its tactile expert (observations then carry `touch` / `touch_mask`, see sim_touch).
The jointly trained policy decoders, LoRA and all other weights come from the checkpoint."""
import argparse
import dataclasses
import hashlib
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, default=18000)
    p.add_argument('--rng-offset', type=int, default=0)
    p.add_argument('--ss-run', type=Path, required=True)
    p.add_argument('--closed-feedback', action='store_true',
                   help='serve block session protocol; requires a jointly trained closed-loop run')
    args = p.parse_args()
    for key in ('checkpoint', 'data', 'base', 'output', 'ss_run'):
        setattr(args, key, getattr(args, key).resolve())
    openpi_root = Path(os.environ['AFCE_OPENPI_ROOT']).resolve()
    sys.path.insert(0, str(openpi_root/'scripts'))
    os.chdir(openpi_root)
    import dexjoco_11_task_balanced as baseline
    from serve_dexjoco_11_task_checkpoint import _load_jax_model
    from openpi.policies.policy import Policy
    from openpi.training import checkpoints
    from openpi import transforms
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from afce_all11.pi_bridge_joint import JointDecodedEffectPolicy, JointEffectWeights
    from afce_ss.ss_model import SingleStageDataConfig, SingleStagePiConfig
    import json
    contract = json.loads((args.ss_run/'training_contract.json').read_text())
    init = args.ss_run/'decoder_init.npz'
    cfg, _, _ = baseline.build_config(baseline.Args(operation='inspect', data_root=args.data, init_params_path=args.base,
        exp_name=args.name, checkpoint_base_dir=args.output/'checkpoints', assets_base_dir=args.output/'assets'))
    cfg = dataclasses.replace(cfg, name='afce_all11_official', data=dataclasses.replace(cfg.data, balance='proportional'))
    trained = contract['model']
    # A run trained with --touch carries scheme B's tactile settings in its model config.
    late_touch = bool(trained.get('late_touch', False))
    closed_loop = 'closed_block_size' in trained
    if closed_loop != args.closed_feedback:
        raise ValueError('Closed-loop checkpoints require --closed-feedback and the block-aware client')
    touch = bool(trained.get('touch_scale'))
    model_class, touch_kw = SingleStagePiConfig, {}
    if late_touch:
        from afce_ss.ss_late_touch import SingleStageLateTouchPiConfig
        model_class = SingleStageLateTouchPiConfig
        touch_kw = {k: trained[k] for k in ('touch_scale', 'late_touch_stats_sha256', 'late_touch_width',
                    'late_touch_limit', 'late_touch_dropout', 'late_touch_mask0')}
        touch_kw['touch_scale'] = tuple(touch_kw['touch_scale'])
        if closed_loop:
            from afce_cl.joint import ClosedLoopPiConfig
            model_class = ClosedLoopPiConfig
            touch_kw.update({k: trained[k] for k in ('closed_block_size', 'closed_width', 'closed_limit')})
    elif touch:
        from afce_ss.ss_touch import SingleStageTouchPiConfig
        model_class = SingleStageTouchPiConfig
        touch_kw = {'touch_scale': tuple(trained['touch_scale']), 'touch_weights_sha256': trained['touch_weights_sha256'],
                    'touch_history': trained['touch_history'], 'touch_dropout': trained['touch_dropout']}
    model = model_class(**dataclasses.asdict(cfg.model), **touch_kw, decoder_init_path=str(init),
        decoder_init_sha256=hashlib.sha256(init.read_bytes()).hexdigest(), decoder_warmup_steps=0,
        alignment_target_mode='ground_truth', alignment_start=trained['alignment_start'],
        alignment_ramp_steps=trained['alignment_ramp_steps'], joint_decoder_lr_scale=trained['joint_decoder_lr_scale'])
    if model.decoder_init_sha256 != trained['decoder_init_sha256']:
        raise ValueError('Decoder initialization differs from the trained run')
    data = SingleStageDataConfig(**{f.name: getattr(cfg.data, f.name) for f in dataclasses.fields(cfg.data)},
                                 afce_cache_root=args.ss_run/'single_stage_cache', touch=touch)
    cfg = dataclasses.replace(cfg, model=model, data=data, weight_loader=JointEffectWeights(str(args.base)))
    jax_model = _load_jax_model(cfg, args.checkpoint)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = checkpoints.load_norm_stats(args.checkpoint/'assets', data.asset_id)
    latent = Policy(jax_model, transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
        transforms.Normalize(norm, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
        output_transforms=[], metadata={'method': 'afce_single_stage_30x256'+('_touch' if touch else ''), 'task': args.task})
    if args.rng_offset < 0:
        raise ValueError('rng-offset must be nonnegative')
    if args.rng_offset:
        import jax
        for _ in range(args.rng_offset):
            latent._rng, _ = jax.random.split(latent._rng)
        print(f'POLICY_RNG_OFFSET requests={args.rng_offset}', flush=True)
    # effect_mean/effect_std are 0/1 in the single-stage model: the policy output is decoded directly.
    if closed_loop:
        from afce_cl.serving import joint_policy
        policy = joint_policy(latent, jax_model, task=args.task)
    elif late_touch:
        from afce_ss.ss_late_touch import LateDecodedEffectPolicy
        policy = LateDecodedEffectPolicy(latent, jax_model, task=args.task)
    else:
        policy = JointDecodedEffectPolicy(latent, jax_model, task=args.task)
    WebsocketPolicyServer(policy=policy, host='0.0.0.0', port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
