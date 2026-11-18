"""One shared block correction for frozen-backbone and joint-training routes."""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def check_block_size(size):
    if not isinstance(size, int) or size < 2 or 30 % size:
        raise ValueError('block_size must divide H30 and be at least 2')
    return size


def indices(offset, size):
    return offset[:, None]+jnp.arange(size)[None]


def take_block(array, offset, size):
    return jnp.take_along_axis(array, indices(offset, size)[..., None], axis=1)


class BlockCorrection(nnx.Module):
    """Complete E and anchored plan + latest state/touch -> next block delta.

    No future-touch targets or partially denoised E. The anchor and current state
    have different projections. Outputs use the original normalized action units.
    """
    def __init__(self, action_dim, *, block_size=5, width=64, limit=.1, rngs):
        self.block_size = check_block_size(block_size)
        if action_dim not in (22, 44) or width < 8 or not np.isfinite(limit) or limit <= 0:
            raise ValueError('Invalid native dimension/width/residual bound')
        self.action_dim, self.state_dim = action_dim, 23*(action_dim//22)
        self.limit, self.width = float(limit), width
        self.e_query = nnx.Linear(256, width, rngs=rngs)
        self.e_memory = nnx.Linear(256, width, rngs=rngs)
        self.plan_in = nnx.Linear(action_dim, width, rngs=rngs)
        self.anchor_in = nnx.Linear(self.state_dim, width, rngs=rngs)
        self.current_in = nnx.Linear(self.state_dim, width, rngs=rngs)
        self.touch_in = nnx.Linear(54, width, rngs=rngs)
        self.touch_hidden = nnx.Linear(width, width, rngs=rngs)
        self.area = nnx.Param(jax.random.normal(rngs(), (8, width))*.02)
        self.position = nnx.Param(jax.random.normal(rngs(), (30, width))*.02)
        self.offset = nnx.Param(jax.random.normal(rngs(), (30//block_size, width))*.02)
        self.touch_key = nnx.Linear(width, width, rngs=rngs)
        self.touch_value = nnx.Linear(width, width, rngs=rngs)
        self.mix = nnx.Linear(3*width, width, rngs=rngs)
        self.out = nnx.Linear(width, action_dim, kernel_init=jax.nn.initializers.zeros,
                              bias_init=jax.nn.initializers.zeros, rngs=rngs)

    def __call__(self, effect, plan, anchor, current, touch, mask, offset, scale):
        b = effect.shape[0]
        if (effect.shape != (b, 30, 256) or plan.shape != (b, 30, self.action_dim)
                or anchor.shape != (b, self.state_dim) or current.shape != anchor.shape
                or touch.shape != (b, 8, 54) or mask.shape != (b, 8)
                or offset.shape != (b,) or scale.shape != (54,)):
            raise ValueError('Closed-loop block tensor contract differs')
        valid_sensors = mask & (jnp.arange(8)[None] < 4*(self.action_dim//22))
        raw = jnp.where(valid_sensors[..., None], touch, 0)
        raw = jnp.sign(raw)*jnp.log1p(jnp.abs(raw))/scale
        tokens = self.touch_hidden(nnx.gelu(self.touch_in(raw)))+self.area.value[None]
        tokens = jnp.where(valid_sensors[..., None], tokens, 0)
        query = (self.e_query(take_block(effect, offset, self.block_size))
                 +self.plan_in(take_block(plan, offset, self.block_size))
                 +self.anchor_in(anchor)[:, None]+self.current_in(current)[:, None]
                 +self.position.value[indices(offset, self.block_size)]
                 +self.offset.value[offset//self.block_size, None])
        attention = jnp.einsum('bkw,bnw->bkn', query, self.touch_key(tokens))/self.width**.5
        weights = jax.nn.softmax(jnp.where(valid_sensors[:, None], attention, -1e30), -1)
        weights = jnp.where(valid_sensors[:, None], weights, 0)
        weights /= jnp.maximum(weights.sum(-1, keepdims=True), 1e-9)
        tactile = jnp.einsum('bkn,bnw->bkw', weights, self.touch_value(tokens))
        memory = self.e_memory(effect)
        e_weights = jax.nn.softmax(jnp.einsum('bkw,bnw->bkn', query, memory)/self.width**.5, -1)
        e_context = jnp.einsum('bkn,bnw->bkw', e_weights, memory)
        delta = self.limit*jnp.tanh(self.out(nnx.gelu(self.mix(jnp.concatenate([query, tactile, e_context], -1)))))
        return jnp.where(valid_sensors.any(-1)[:, None, None], delta, 0)


def block_loss(prediction, truth, valid, action_dim):
    """Equal arm/group weighting: xyz, rotvec, finger; valid pairs stay in block."""
    if action_dim not in (22, 44):
        raise ValueError('Expected native 22D or 44D action')
    groups = []
    for arm in range(action_dim//22):
        for lo, hi in ((0, 3), (3, 6), (6, 22)):
            error = prediction[..., arm*22+lo:arm*22+hi]-truth[..., arm*22+lo:arm*22+hi]
            value = jnp.where(jnp.abs(error) < 1, .5*error**2, jnp.abs(error)-.5)
            weights = jnp.broadcast_to(valid[..., None], value.shape)
            groups.append(jnp.where(weights, value, 0).sum()/jnp.maximum(weights.sum(), 1))
    action = jnp.stack(groups).mean()
    pf = jnp.concatenate([prediction[..., a*22+6:(a+1)*22] for a in range(action_dim//22)], -1)
    tf = jnp.concatenate([truth[..., a*22+6:(a+1)*22] for a in range(action_dim//22)], -1)
    target_delta = tf[:, 1:]-tf[:, :-1]
    delta = (pf[:, 1:]-pf[:, :-1])-target_delta
    pairs = jnp.broadcast_to((valid[:, 1:] & valid[:, :-1])[..., None], delta.shape)
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.sqrt(
        jnp.where(pairs, target_delta**2, 0).sum((0, 1))/jnp.maximum(pairs.sum((0, 1)), 1)+1e-8), .1))
    delta /= scale
    value = jnp.where(jnp.abs(delta) < 1, .5*delta**2, jnp.abs(delta)-.5)
    finger_delta = jnp.where(pairs, value, 0).sum()/jnp.maximum(pairs.sum(), 1)
    return .25628781345139295*action+.05*finger_delta, {'action_loss': action, 'finger_delta_loss': finger_delta}
