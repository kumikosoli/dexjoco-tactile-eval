"""Fusion scheme: the C01 single-finger Effect policy with FTP-1's pretrained tactile expert (scheme B) and
each fingertip's recent touch history.

The image-language prefix never sees touch. The action expert, which predicts E, additionally attends to the
tactile block (one token per fingertip: its current values plus an encoding of its last 0.9 s). Token positions
follow FTP-1: prefix, then tactile, then the action block. With no valid tactile token the computation is the
original policy's; in training the tactile block of a sample is dropped with probability `touch_dropout`, so
the same checkpoint can also be run without touch. E's normalization, the decoders and every loss are
unchanged."""
import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_all11.pi_bridge_joint import JointEffectWeights
from afce_all11.single_finger_alignment import SingleFingerEffectPi, SingleFingerEffectPiConfig
from afce_touch.expert import TouchExpert, convert
from afce_touch.layout import HISTORY_FRAMES, PACKED
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at
from openpi.training import weight_loaders


class TouchEffectPi(SingleFingerEffectPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        # Created last: every original module is initialised exactly as in the policy without touch.
        self.touch = TouchExpert(config.dtype, rngs, config.touch_history)
        self.touch_scale = nnx.Variable(jnp.asarray(config.touch_scale, jnp.float32))
        self.touch_history = config.touch_history
        self.touch_dropout = config.touch_dropout

    def touch_cache(self, observation, prefix_mask, drop_rng=None):
        if observation.touch is None or observation.touch_mask is None:
            raise ValueError('The tactile policy requires the tactile tokens')
        mask = observation.touch_mask
        if drop_rng is not None and self.touch_dropout > 0:
            mask = mask & (jax.random.uniform(drop_rng, mask.shape[:1]) >= self.touch_dropout)[:, None]
        packed = observation.touch.astype(jnp.float32)
        scale = self.touch_scale.value
        values = jnp.sign(packed[..., :54])*jnp.log1p(jnp.abs(packed[..., :54]))/scale
        history = None
        if self.touch_history:
            frames = packed[..., 54:].reshape(*packed.shape[:-1], HISTORY_FRAMES, 7)
            summary = jnp.sign(frames[..., :6])*jnp.log1p(jnp.abs(frames[..., :6]))/scale[48:54]
            change = jnp.concatenate([jnp.zeros_like(summary[..., :1, :]), summary[..., 1:, :]-summary[..., :-1, :]], -2)
            history = jnp.concatenate([summary, change, jnp.log1p(frames[..., 6:])/jnp.log1p(50.)], -1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(mask, axis=1)-1
        return self.touch.cache(self.touch.tokens(values, mask, history), mask, positions), mask

    def flow(self, rng, observation, train):
        """ReadoutEffectPi.flow with the tactile block's keys and values in front of the joint sequence."""
        target = jax.lax.stop_gradient(observation.effect_target)
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        given = observation  # preprocess_observation rebuilds the observation without the tactile fields
        observation = model_lib.preprocess_observation(pre_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, target.shape)
        time = jax.random.beta(time_rng, 1.5, 1, target.shape[:-2])*.999+.001
        t = time[..., None, None]
        noisy = t*noise+(1-t)*target
        truth_velocity = noise-target
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, noisy, time)
        # A separate key: the noise and time draws are those of the policy without touch.
        cache, touch_mask = self.touch_cache(given, prefix_mask, jax.random.fold_in(rng, 7) if train else None)
        mask = jnp.concatenate([prefix_mask, suffix_mask], 1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], 0)
        in_suffix = jnp.concatenate([jnp.zeros_like(prefix_mask), suffix_mask], 1)
        to_touch = in_suffix[:, :, None] & touch_mask[:, None, :]
        attention = jnp.concatenate([to_touch, make_attn_mask(mask, ar)], axis=2)  # cached keys come first
        before = jnp.sum(prefix_mask, axis=-1)[:, None]
        positions = jnp.concatenate([jnp.cumsum(prefix_mask, axis=1)-1,
                                     before+jnp.sum(touch_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=1)-1], 1)
        (_, suffix_out), _ = self.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions,
                                                adarms_cond=[None, condition], kv_cache=cache)
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:]).astype(jnp.float32)
        fm = jnp.mean(jnp.square(velocity-truth_velocity))
        return fm, noisy-t*velocity, target, velocity, truth_velocity, time

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        """Pi0.sample_actions with the tactile cache appended to the prefix cache."""
        given = observation
        observation = model_lib.preprocess_observation(None, observation, train=False)
        dt = -1.0/num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1)-1
        _, prefix_cache = self.PaliGemma.llm([prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar_mask),
                                             positions=positions)
        touch_cache, touch_mask = self.touch_cache(given, prefix_mask)
        kv_cache = tuple(jnp.concatenate([p, c.astype(p.dtype)], axis=2) for p, c in zip(prefix_cache, touch_cache, strict=True))
        context_mask = jnp.concatenate([prefix_mask, touch_mask], axis=1)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            context_attn_mask = einops.repeat(context_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([context_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(context_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=-1)-1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond])
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t+dt*v_t, time+dt

        def cond(carry):
            _x_t, time = carry
            return time >= -dt/2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


@dataclasses.dataclass(frozen=True)
class TouchEffectPiConfig(SingleFingerEffectPiConfig):
    touch_scale: tuple = ()
    touch_weights_sha256: str = ''
    touch_history: bool = True
    touch_dropout: float = .1
    touch_design: str = ('FTP-1 v0426 50k tactile expert, area embedding and unified projection; one token per fingertip in '
                         'FTP-1 area order: current values through a tokenizer from scratch plus, with touch_history, a '
                         'two-layer temporal encoder (from scratch, zero-initialised output) of its last 16 summaries at '
                         '60 ms spacing; action expert attends to it; the block is dropped per sample with touch_dropout')

    def create(self, rng):
        return TouchEffectPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation = dataclasses.replace(observation,
                touch=jax.ShapeDtypeStruct((batch_size, 8, PACKED), jnp.float32),
                touch_mask=jax.ShapeDtypeStruct((batch_size, 8), jnp.bool_))
        return observation, actions


@dataclasses.dataclass(frozen=True)
class TouchWeights:
    params_path: str
    touch_weights_path: str

    def load(self, params):
        loaded = model_lib.restore_params(self.params_path, restore_type=np.ndarray)
        loaded = {name: value for name, value in loaded.items() if name not in ('action_in_proj', 'action_out_proj')}
        with np.load(self.touch_weights_path, allow_pickle=False) as z:
            pretrained = convert({name: z[name] for name in z.files})
        loaded['touch'] = pretrained
        return weight_loaders._merge_params(loaded, params, missing_regex=(
            r'.*(lora|action_in_proj|action_out_proj|single_decoder|bimanual_decoder|effect_mean|effect_std'
            r'|touch/tokenizer_in|touch/tokenizer_out|touch/history|touch_scale).*'))


def touch_config_from_environment(config):
    """AFCE_TOUCH_ROOT: the tactile feature cache (its tactile_stats.json gives the scale);
    AFCE_TOUCH_WEIGHTS: the FTP-1 tensors extracted by extract_ftp.py; AFCE_TOUCH_HISTORY=0 turns the history
    encoder off (scheme B's token) and AFCE_TOUCH_DROPOUT sets the training drop probability."""
    import hashlib
    import json
    import os
    from pathlib import Path
    root, weights = Path(os.environ['AFCE_TOUCH_ROOT']), Path(os.environ['AFCE_TOUCH_WEIGHTS'])
    scale = json.loads((root/'tactile_stats.json').read_text())['scale']
    receipt = json.loads(weights.with_suffix('.json').read_text())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert receipt['sha256'] == digest, 'Extracted FTP-1 tensors changed'
    return touch_config(config, scale, weights, digest, os.environ.get('AFCE_TOUCH_HISTORY', '1') == '1',
                        float(os.environ.get('AFCE_TOUCH_DROPOUT', '.1')))


def touch_config(config, scale, touch_weights, touch_weights_sha256, history=True, dropout=.1):
    assert isinstance(config.weight_loader, JointEffectWeights) and len(scale) == 54
    model = TouchEffectPiConfig(**dataclasses.asdict(config.model), touch_scale=tuple(float(x) for x in scale),
                                touch_weights_sha256=touch_weights_sha256, touch_history=history, touch_dropout=dropout)
    return dataclasses.replace(config, model=model,
                               weight_loader=TouchWeights(config.weight_loader.params_path, str(touch_weights)))
