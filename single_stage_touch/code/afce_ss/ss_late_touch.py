"""Single-stage AFCE with E and current touch meeting ONLY at action decoding.

No tactile conditioning of the E teacher or pi's flow. No future-touch loss.
Clean EMA-E decoding remains touch-free. First experiment: one touch observation
per policy request, complete 30-action chunks; NOT yet TacForcing streaming.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_ss.late_touch import LateTouchCorrection, prepare_current_touch, validate_scale
from afce_ss.ss_model import SingleStagePi, SingleStagePiConfig, policy_action_loss
from openpi.models import model as model_lib
from openpi.shared import array_typing as at
from openpi.training import weight_loaders


class SingleStageLateTouchPi(SingleStagePi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        if self.decoupled_aux:
            raise ValueError('Late-touch v1 supports the unchanged v1 auxiliary gradient path only')
        validate_scale(config.touch_scale)
        self.late_touch_scale = nnx.Variable(jnp.asarray(config.touch_scale, jnp.float32))
        self.late_touch_single = LateTouchCorrection(22, width=config.late_touch_width,
            residual_limit=config.late_touch_limit, rngs=rngs)
        self.late_touch_bimanual = LateTouchCorrection(44, width=config.late_touch_width,
            residual_limit=config.late_touch_limit, rngs=rngs)
        self.late_touch_dropout = config.late_touch_dropout
        self.late_touch_mask0 = config.late_touch_mask0

    def decode_with_touch(self, effect, state, touch, touch_mask, action_dim):
        """Complete E + ORIGINAL raw anchor state + current touch -> normalized actions."""
        decoder = self.single_decoder if action_dim == 22 else self.bimanual_decoder
        correction = self.late_touch_single if action_dim == 22 else self.late_touch_bimanual
        planned = decoder(effect, state)
        state_n = (state-decoder.state_mean.value)/decoder.state_std.value
        if self.late_touch_mask0:
            touch_mask = jnp.zeros_like(touch_mask, dtype=jnp.bool_)
        return planned+correction(effect, state_n, planned, touch, touch_mask, self.late_touch_scale.value)

    def _predictions(self, estimate, observation):
        if observation.touch is None or observation.touch_mask is None:
            raise ValueError('Late fusion requires current touch and a sensor-presence mask')
        effect = estimate*self.effect_std.value+self.effect_mean.value
        for decoder, dim, sd in ((self.single_decoder, 22, 23), (self.bimanual_decoder, 44, 46)):
            prediction = self.decode_with_touch(effect, observation.effect_state[:, :sd],
                observation.touch, observation.touch_mask, dim)
            truth = jax.lax.stop_gradient((observation.effect_actions[..., :dim]-decoder.action_mean.value)
                                          /decoder.action_std.value)
            yield decoder, dim, prediction, truth

    def auxiliary(self, estimate, reference, observation, weights, mode):
        if mode != 'ground_truth':
            raise ValueError('Late fusion preserves GT-action supervision')
        losses = [policy_action_loss(pred, truth, observation.effect_valid, dim, decoder, self.rotation_loss)
                  for decoder, dim, pred, truth in self._predictions(estimate, observation)]
        single = observation.effect_action_dim == 22
        values = jnp.where(single, losses[0], losses[1])
        return jnp.mean(weights*values), (values, single)

    def finger_delta_loss(self, estimate, observation, weights):
        # Unchanged C01 valid-pair, scale-floor and grouping rules, now using late-fused actions.
        losses = []
        for decoder, dim, prediction, truth in self._predictions(estimate, observation):
            slices = ((6, 22),) if dim == 22 else ((6, 22), (28, 44))
            pf = jnp.concatenate([prediction[..., lo:hi] for lo, hi in slices], -1)
            tf = jnp.concatenate([truth[..., lo:hi] for lo, hi in slices], -1)
            pd, td = pf[:, 1:]-pf[:, :-1], tf[:, 1:]-tf[:, :-1]
            valid = observation.effect_valid[:, 1:] & observation.effect_valid[:, :-1]
            valid = valid & (observation.effect_action_dim == dim)[:, None]
            mask = jnp.broadcast_to(valid[..., None], td.shape)
            scale = jax.lax.stop_gradient(jnp.maximum(jnp.sqrt(
                jnp.sum(jnp.where(mask, td**2, 0), (0, 1), keepdims=True)
                /jnp.maximum(mask.sum((0, 1), keepdims=True), 1)+1e-8), self.finger_delta_scale_floor))
            error = jnp.abs((pd-td)/scale)
            huber = jnp.where(error < 1, .5*error**2, error-.5)
            losses.append(jnp.sum(jnp.where(mask, huber, 0), (1, 2))/jnp.maximum(mask.sum((1, 2)), 1))
        values = jnp.where(observation.effect_action_dim == 22, losses[0], losses[1])
        return jnp.mean(weights*values), values

    def compute_loss_and_metrics_at_step(self, rng, observation, actions, *, step, train=False):
        if observation.touch_mask is None:
            raise ValueError('Missing current tactile sensor mask')
        mask = observation.touch_mask
        if train and self.late_touch_dropout:
            keep = jax.random.uniform(jax.random.fold_in(rng, 71), mask.shape[:1]) >= self.late_touch_dropout
            mask = mask & keep[:, None]
        if self.late_touch_mask0:
            mask = jnp.zeros_like(mask, dtype=jnp.bool_)
        observation = dataclasses.replace(observation, touch_mask=mask)
        # The inherited flow never reads touch; clean_decode_loss still calls the original decoders directly.
        loss, metrics = super().compute_loss_and_metrics_at_step(rng, observation, actions, step=step, train=train)
        metrics['late_touch_present_fraction'] = mask.any(-1).astype(jnp.float32).mean()
        return loss, metrics


@dataclasses.dataclass(frozen=True)
class SingleStageLateTouchPiConfig(SingleStagePiConfig):
    joint_objective: str = 'afce_single_stage_c01_late_current_touch_v1'
    late_touch: bool = True
    touch_scale: tuple = ()
    late_touch_stats_sha256: str = ''
    late_touch_width: int = 64
    late_touch_limit: float = .1
    late_touch_dropout: float = 0.
    late_touch_mask0: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.late_touch_dropout < 1:
            raise ValueError('late-touch dropout must be in [0,1)')

    def create(self, rng):
        return SingleStageLateTouchPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = dataclasses.replace(obs, touch=jax.ShapeDtypeStruct((batch_size, 8, 166), jnp.float32),
                                      touch_mask=jax.ShapeDtypeStruct((batch_size, 8), jnp.bool_))
        return obs, actions


@dataclasses.dataclass(frozen=True)
class LateTouchWeights:
    params_path: str

    def load(self, params):
        loaded = model_lib.restore_params(self.params_path, restore_type=np.ndarray)
        loaded = {k: v for k, v in loaded.items() if k not in ('action_in_proj', 'action_out_proj')}
        return weight_loaders._merge_params(loaded, params, missing_regex=(
            r'.*(lora|action_in_proj|action_out_proj|single_decoder|bimanual_decoder|effect_mean|effect_std|late_touch).*'))


def late_touch_settings(*, width=64, limit=.1, dropout=0., mask0=False):
    path = Path(os.environ['AFCE_TOUCH_ROOT'])/'tactile_stats.json'
    scale = validate_scale(json.loads(path.read_text())['scale'])
    return dict(touch_scale=tuple(float(x) for x in scale), late_touch_width=width, late_touch_limit=limit,
                late_touch_dropout=dropout, late_touch_mask0=mask0,
                late_touch_stats_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class LateDecodedEffectPolicy:
    """First experiment's stateless, per-request late fusion. No hidden action queue or streaming claim."""
    def __init__(self, latent_policy, model, *, task):
        self.policy, self.model = latent_policy, model
        self.state_dim = 46 if task.startswith('bimanual_') else 23
        self.action_dim = 44 if self.state_dim == 46 else 22
        decoder = model.bimanual_decoder if self.action_dim == 44 else model.single_decoder
        self.decode = jax.jit(lambda e, s, t, m: (
            model.decode_with_touch(e*model.effect_std.value+model.effect_mean.value, s, t, m, self.action_dim)
            *decoder.action_std.value+decoder.action_mean.value))

    @property
    def metadata(self):
        return {**self.policy.metadata, 'touch_fusion': 'decoder_only', 'touch_refresh': 'per_request',
                'streaming': False, 'future_touch_prediction': False}

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def infer(self, observation, **kwargs):
        state = np.asarray(observation['state'], dtype=np.float32)[:self.state_dim]
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError('Decoder requires finite raw native23/46 anchor state')
        touch, mask = prepare_current_touch(observation['touch'], observation['touch_mask'])
        started = time.monotonic()
        result = dict(self.policy.infer(observation, **kwargs))
        e = np.asarray(result['actions'], dtype=np.float32)
        if e.shape != (30, 256) or not np.isfinite(e).all():
            raise ValueError('Late fusion requires a complete finite 30x256 E')
        actions = np.asarray(self.decode(e[None], state[None], touch[None], mask[None]))[0]
        if actions.shape != (30, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError('Invalid late-fusion physical action chunk')
        result['actions'] = actions
        result['policy_timing'] = {'infer_ms': 1000*(time.monotonic()-started)}
        return result
