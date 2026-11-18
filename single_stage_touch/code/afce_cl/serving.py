"""Shared block-feedback policy for light or jointly trained correction weights."""
from __future__ import annotations
import json
from pathlib import Path
import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_cl.io import sha, source_identity
from afce_cl.light import CorrectionPair
from afce_cl.session import BlockSession
from afce_ss.late_touch import prepare_current_touch


class FeedbackPolicy:
    def __init__(self, latent, model, correction, *, task, block_size):
        self.latent, self.model = latent, model
        self.dim = 44 if task.startswith('bimanual_') else 22
        self.sd, self.task = 23*(self.dim//22), task
        decoder = model.single_decoder if self.dim == 22 else model.bimanual_decoder
        self.mean, self.std = np.asarray(decoder.action_mean.value), np.asarray(decoder.action_std.value)
        sm, ss = decoder.state_mean.value, decoder.state_std.value
        self.decode = jax.jit(lambda e, state, touch, mask:
            model.decode_with_touch(e, state, touch, mask, self.dim))
        self.correction = jax.jit(lambda e, plan, anchor, current, touch, mask, offset:
            correction(e, plan, (anchor-sm)/ss, (current-sm)/ss, touch, mask, offset, model.late_touch_scale.value))
        self.session = BlockSession(self.make_plan, self.correct_block, action_dim=self.dim, block_size=block_size)

    @property
    def metadata(self):
        return {'method': 'afce_complete_E_block_feedback_v1', 'block_size': self.session.block_size,
            'action_horizon': 30, 'task': self.task, 'touch_refresh': 'each_acknowledged_block',
            'future_touch_prediction': False, 'streaming_E': False}

    def reset(self):
        self.session.reset()

    def make_plan(self, observation):
        state = np.asarray(observation['state'], np.float32)[:self.sd]
        touch, mask = prepare_current_touch(observation['touch'], observation['touch_mask'])
        sampled = np.asarray(self.latent.infer(observation)['actions'], np.float32)
        if sampled.shape != (30, 256):raise ValueError('Expected complete sampled E')
        effect = sampled*np.asarray(self.model.effect_std.value)+np.asarray(self.model.effect_mean.value)
        planned = np.asarray(self.decode(effect[None], state[None], touch[None], mask[None]))[0]
        return effect, planned*self.std+self.mean

    def correct_block(self, effect, plan, anchor, current, touch, mask, offset):
        normalized = (plan-self.mean)/self.std
        delta = np.asarray(self.correction(effect[None], normalized[None], anchor[None], current[None],
            touch[None], mask[None], np.asarray([offset], np.int32)))[0]
        return (normalized[offset:offset+self.session.block_size]+delta)*self.std+self.mean

    def infer(self, observation, **kwargs):
        return self.session.infer(observation)


def light_policy(run, correction_run, data, output, *, task):
    from afce_cl.frozen import load_frozen
    run, correction_run = Path(run), Path(correction_run)
    receipt = json.loads((correction_run/'latest.json').read_text())
    if not receipt['complete'] or sha(correction_run/'latest.pkl') != receipt['sha256']:
        raise ValueError('Incomplete/corrupt light correction checkpoint')
    with (correction_run/'latest.pkl').open('rb') as f:state = pickle.load(f)
    contract, config = state['contract'], state['contract']['config']
    if source_identity(run)['source_fingerprint'] != contract['source']['source_fingerprint']:
        raise ValueError('Correction was trained against another frozen source model')
    pair = CorrectionPair(block_size=config['block_size'], width=config['width'], limit=config['limit'], seed=config['seed'])
    # Checkpoints intentionally contain host arrays. Dynamic JIT indexing of
    # positional weights requires restoring them to JAX arrays before tracing.
    nnx.update(pair, jax.tree.map(jnp.asarray, state['optimizer']['model']))
    latent, model = load_frozen(run, data, output)
    module = pair.bimanual if task.startswith('bimanual_') else pair.single
    return FeedbackPolicy(latent, model, module, task=task, block_size=config['block_size'])


def joint_policy(latent, model, *, task):
    module = model.closed_loop_bimanual if task.startswith('bimanual_') else model.closed_loop_single
    return FeedbackPolicy(latent, model, module, task=task, block_size=model.closed_block_size)
