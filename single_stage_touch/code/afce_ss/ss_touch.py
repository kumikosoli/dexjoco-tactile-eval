"""Single-stage AFCE policy with scheme B's tactile input (FTP-1's pretrained tactile expert, current touch).

SingleStagePi with afce_touch.policy.TouchEffectPi's tactile path: the image-language prefix never sees touch
(so the pooled prefix feature of the contrastive loss is unchanged); the action expert, which predicts E, also
attends to one token per fingertip of the current control interval. In training the tactile block of a sample
is dropped with probability touch_dropout (a key of its own: the noise and time draws are those of the policy
without touch). E, the encoder, the decoders and every loss are the single-stage run's."""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from afce_ss.ss_model import SingleStagePi, SingleStagePiConfig
from afce_touch.expert import TouchExpert
from afce_touch.layout import PACKED
from afce_touch.policy import TouchEffectPi, TouchWeights
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at


class SingleStageTouchPi(SingleStagePi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        if self.effect_prediction != 'velocity':
            raise ValueError('The tactile single-stage policy uses the velocity parameterization')
        # Created last: every original module is initialised exactly as in the policy without touch.
        self.touch = TouchExpert(config.dtype, rngs, config.touch_history)
        self.touch_scale = nnx.Variable(jnp.asarray(config.touch_scale, jnp.float32))
        self.touch_history = config.touch_history
        self.touch_dropout = config.touch_dropout

    touch_cache = TouchEffectPi.touch_cache
    sample_actions = TouchEffectPi.sample_actions

    def flow_with_prefix(self, rng, observation, train):
        """SingleStagePi.flow_with_prefix with the tactile block's keys and values in front of the joint sequence."""
        target = jax.lax.stop_gradient(observation.effect_target)
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        given = observation  # preprocess_observation rebuilds the observation without the tactile fields
        observation = model_lib.preprocess_observation(pre_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, target.shape)
        time = self.sample_time(time_rng, target.shape[:-2])
        t = time[..., None, None]
        noisy = t*noise+(1-t)*target
        truth_velocity = noise-target
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, noisy, time)
        cache, touch_mask = self.touch_cache(given, prefix_mask, jax.random.fold_in(rng, 7) if train else None)
        mask = jnp.concatenate([prefix_mask, suffix_mask], 1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], 0)
        in_suffix = jnp.concatenate([jnp.zeros_like(prefix_mask), suffix_mask], 1)
        to_touch = in_suffix[:, :, None] & touch_mask[:, None, :]
        attention = jnp.concatenate([to_touch, make_attn_mask(mask, ar)], axis=2)  # cached keys come first
        before = jnp.sum(prefix_mask, axis=-1)[:, None]
        positions = jnp.concatenate([jnp.cumsum(prefix_mask, axis=1)-1,
                                     before+jnp.sum(touch_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=1)-1], 1)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions,
                                                         adarms_cond=[None, condition], kv_cache=cache)
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:]).astype(jnp.float32)
        if self.fm_time_weight == 'none':
            fm = jnp.mean(jnp.square(velocity-truth_velocity))
        else:
            fm = jnp.mean(jnp.square(velocity-truth_velocity)*self.fm_sample_weight(time)[..., None, None])
        weights = prefix_mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(prefix_out.astype(jnp.float32)*weights, 1)/jnp.maximum(weights.sum(1), 1)
        return fm, noisy-t*velocity, target, velocity, truth_velocity, time, jax.lax.stop_gradient(pooled)


@dataclasses.dataclass(frozen=True)
class SingleStageTouchPiConfig(SingleStagePiConfig):
    joint_objective: str = 'afce_single_stage_c01_contrastive_v1_touch'
    touch_scale: tuple = ()
    touch_weights_sha256: str = ''
    touch_history: bool = False
    touch_dropout: float = .1
    touch_design: str = ('scheme B: FTP-1 v0426 50k tactile expert, area embedding and unified projection; one token per '
                         'fingertip of the current control interval in FTP-1 area order; tokenizer from scratch; action '
                         'expert attends to it; the block is dropped per training sample with touch_dropout')

    def create(self, rng):
        return SingleStageTouchPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation = dataclasses.replace(observation,
                touch=jax.ShapeDtypeStruct((batch_size, 8, PACKED), jnp.float32),
                touch_mask=jax.ShapeDtypeStruct((batch_size, 8), jnp.bool_))
        return observation, actions


def touch_settings(dropout):
    """Scale and FTP-1 tensors from AFCE_TOUCH_ROOT / AFCE_TOUCH_WEIGHTS (as afce_touch.policy), history off."""
    import hashlib
    import json
    import os
    from pathlib import Path
    root, weights = Path(os.environ['AFCE_TOUCH_ROOT']), Path(os.environ['AFCE_TOUCH_WEIGHTS'])
    scale = json.loads((root/'tactile_stats.json').read_text())['scale']
    receipt = json.loads(weights.with_suffix('.json').read_text())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert receipt['sha256'] == digest, 'Extracted FTP-1 tensors changed'
    assert len(scale) == 54
    return {'touch_scale': tuple(float(x) for x in scale), 'touch_weights_sha256': digest,
            'touch_history': False, 'touch_dropout': float(dropout)}, str(weights)


def touch_weight_loader(base_params, weights):
    return TouchWeights(str(base_params), weights)
