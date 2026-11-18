"""Small, decoder-side current-touch branch; no FTP dependency or future-touch targets.

All outputs are residuals in the ORIGINAL decoder's normalized action coordinates.
E is a complete plan; this module does not implement streaming/partially denoised E.
"""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

CURRENT_FEATURES = 54
TOUCH_SLOTS = 8
HORIZON = 30


def prepare_current_touch(values, mask):
    """Validate an unbatched observation, accepting the existing packed-history wire format.

    Only the first 54 CURRENT values are consumed. Missing sensors are distinguished
    from valid zero-contact values exclusively through the boolean mask.
    """
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask)
    if values.shape not in ((8, 54), (8, 166)) or mask.shape != (8,) or mask.dtype != np.bool_:
        raise ValueError('Expected touch (8,54) or existing (8,166), and boolean touch_mask (8,)')
    current = values[:, :54].copy()
    if not np.isfinite(current[mask]).all():
        raise ValueError('Nonfinite values in a present tactile sensor')
    return np.where(mask[:, None], current, 0), mask.copy()


def validate_scale(scale):
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (54,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Current-touch scale must contain 54 finite positive values')
    return scale


class LateTouchCorrection(nnx.Module):
    """E/state/planned-action queries attend to separate CURRENT fingertip tokens.

    A zero-initialized head preserves the original decoder at initialization.
    Missing touch gives an exact zero residual, including after training.
    Single-arm models never use the four padded left-hand slots.
    """
    def __init__(self, action_dim, *, width=64, residual_limit=.1, rngs):
        if action_dim not in (22, 44) or width < 8 or not np.isfinite(residual_limit) or residual_limit <= 0:
            raise ValueError('Expected action_dim 22/44, width >= 8, positive residual_limit')
        self.action_dim = action_dim
        self.state_dim = 23*(action_dim//22)
        self.width = width
        self.residual_limit = float(residual_limit)
        self.effect_in = nnx.Linear(256, width, rngs=rngs)
        self.state_in = nnx.Linear(self.state_dim, width, rngs=rngs)
        self.action_in = nnx.Linear(action_dim, width, rngs=rngs)
        self.touch_in = nnx.Linear(54, width, rngs=rngs)
        self.touch_out = nnx.Linear(width, width, rngs=rngs)
        self.area = nnx.Param(jax.random.normal(rngs(), (8, width))*.02)
        self.positions = nnx.Param(jax.random.normal(rngs(), (30, width))*.02)
        self.key = nnx.Linear(width, width, rngs=rngs)
        self.value = nnx.Linear(width, width, rngs=rngs)
        self.mix = nnx.Linear(2*width, width, rngs=rngs)
        self.out = nnx.Linear(width, action_dim, kernel_init=jax.nn.initializers.zeros,
                              bias_init=jax.nn.initializers.zeros, rngs=rngs)

    def __call__(self, effect, state_normalized, planned_normalized, touch, touch_mask, scale):
        b = effect.shape[0]
        if (effect.shape != (b, 30, 256) or state_normalized.shape != (b, self.state_dim)
                or planned_normalized.shape != (b, 30, self.action_dim)
                or touch.ndim != 3 or touch.shape[:2] != (b, 8) or touch.shape[-1] not in (54, 166)
                or touch_mask.shape != (b, 8) or scale.shape != (54,)):
            raise ValueError('E/state/action/touch shape differs from the native C01 late-fusion contract')
        mask = touch_mask.astype(jnp.bool_) & (jnp.arange(8)[None] < 4*(self.action_dim//22))
        # Mask BEFORE normalization; absent/padded sensors may contain arbitrary data.
        current = jnp.where(mask[..., None], touch[..., :54], 0).astype(jnp.float32)
        current = jnp.sign(current)*jnp.log1p(jnp.abs(current))/scale
        tokens = self.touch_out(nnx.gelu(self.touch_in(current)))+self.area.value[None]
        tokens = jnp.where(mask[..., None], tokens, 0)
        q = (self.effect_in(effect)+self.state_in(state_normalized)[:, None]
             +self.action_in(planned_normalized)+self.positions.value[None])
        logits = jnp.einsum('bhw,bnw->bhn', q, self.key(tokens))/self.width**.5
        weights = jax.nn.softmax(jnp.where(mask[:, None], logits, -1e30), axis=-1)
        weights = jnp.where(mask[:, None], weights, 0)
        weights = weights/jnp.maximum(weights.sum(-1, keepdims=True), 1e-9)
        context = jnp.einsum('bhn,bnw->bhw', weights, self.value(tokens))
        residual = self.residual_limit*jnp.tanh(self.out(nnx.gelu(self.mix(jnp.concatenate([q, context], -1)))))
        return jnp.where(mask.any(-1)[:, None, None], residual, 0)
