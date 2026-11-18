"""Joint single-stage training with complete E and causally aligned block feedback."""
from __future__ import annotations
import dataclasses
import os
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_cl.block import BlockCorrection, check_block_size, indices
from afce_cl.timeline import Timeline
from afce_ss.ss_data import SingleStageDataset
from afce_ss.ss_late_touch import SingleStageLateTouchPi, SingleStageLateTouchPiConfig
from afce_ss.ss_model import SingleStageDataConfig, PreserveTargets, policy_action_loss
from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import array_typing as at
from openpi.training import weight_loaders

FIELDS = ('closed_state', 'closed_touch', 'closed_touch_mask', 'closed_offset')


class ClosedLoopDataset(SingleStageDataset):
    def __init__(self, dataset, root):
        super().__init__(dataset, root)
        self.timeline = None

    def __getstate__(self):
        state = super().__getstate__()
        state['timeline'] = None
        return state

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.timeline is None:
            self.timeline = Timeline(self.data_root, os.environ['AFCE_TOUCH_ROOT'])
        size = check_block_size(int(os.environ['AFCE_CL_BLOCK_SIZE']))
        ep, frame = int(sample['episode_index']), int(sample['frame_index'])
        n = self.inputs.length(ep)
        choices = [k for k in range(0, 30, size) if frame+k < n]
        offset = int(np.random.choice(choices))  # inside upstream per-visit RNG context
        sample.update(self.timeline.feedback(self.task, ep, frame, offset, size))
        return sample


@dataclasses.dataclass(frozen=True)
class ClosedLoopDataConfig(SingleStageDataConfig):
    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        repack = base.repack_transforms.inputs[0]
        inner = base.data_transforms.inputs[0]
        return dataclasses.replace(base,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(
                {**repack.structure, **{name: name for name in FIELDS}})]),
            data_transforms=dataclasses.replace(base.data_transforms,
                inputs=[PreserveTargets(inner.inputs, inner.names+FIELDS)]))


class ClosedLoopPi(SingleStageLateTouchPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.closed_block_size = config.closed_block_size
        self.closed_loop_single = BlockCorrection(22, block_size=config.closed_block_size,
            width=config.closed_width, limit=config.closed_limit, rngs=rngs)
        self.closed_loop_bimanual = BlockCorrection(44, block_size=config.closed_block_size,
            width=config.closed_width, limit=config.closed_limit, rngs=rngs)

    def feedback_prediction(self, effect, anchor, current, touch, mask, offset, planned, dim):
        decoder = self.single_decoder if dim == 22 else self.bimanual_decoder
        block = self.closed_loop_single if dim == 22 else self.closed_loop_bimanual
        norm = lambda x: (x-decoder.state_mean.value)/decoder.state_std.value
        correction = block(effect, planned, norm(anchor), norm(current), touch, mask, offset,
                           self.late_touch_scale.value)
        result = planned.at[jnp.arange(len(planned))[:, None], indices(offset, self.closed_block_size)].add(correction)
        return result

    def _predictions(self, estimate, observation):
        if any(getattr(observation, k) is None for k in FIELDS):
            raise ValueError('Joint block training needs aligned current feedback fields')
        effect = estimate*self.effect_std.value+self.effect_mean.value
        for decoder, dim, sd in ((self.single_decoder, 22, 23), (self.bimanual_decoder, 44, 46)):
            # Anchor touch/state produce a fixed chunk plan; feedback touches only the selected block.
            planned = self.decode_with_touch(effect, observation.effect_state[:, :sd],
                observation.touch, observation.touch_mask, dim)
            prediction = self.feedback_prediction(effect, observation.effect_state[:, :sd],
                observation.closed_state[:, :sd], observation.closed_touch, observation.closed_touch_mask,
                observation.closed_offset, planned, dim)
            truth = jax.lax.stop_gradient((observation.effect_actions[..., :dim]-decoder.action_mean.value)
                                          /decoder.action_std.value)
            yield decoder, dim, prediction, truth

    def selected_valid(self, obs):
        position = jnp.arange(30)[None]
        return obs.effect_valid & (position >= obs.closed_offset[:, None]) & (
            position < obs.closed_offset[:, None]+self.closed_block_size)

    def auxiliary(self, estimate, reference, observation, weights, mode):
        if mode != 'ground_truth':
            raise ValueError('Closed-loop v1 requires GT-action supervision')
        valid = self.selected_valid(observation)
        values = [policy_action_loss(p, t, valid, dim, dec, self.rotation_loss)
                  for dec, dim, p, t in self._predictions(estimate, observation)]
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single, values[0], values[1])
        return jnp.mean(weights*per_sample), (per_sample, single)

    def finger_delta_loss(self, estimate, observation, weights):
        # Reuse the exact parent finger normalization; pairs outside this block are invalid.
        return super().finger_delta_loss(estimate,
            dataclasses.replace(observation, effect_valid=self.selected_valid(observation)), weights)

    def compute_loss_and_metrics_at_step(self, rng, observation, actions, *, step, train=False):
        loss, metrics = super().compute_loss_and_metrics_at_step(rng, observation, actions, step=step, train=train)
        metrics['closed_offset_mean'] = observation.closed_offset.mean()
        metrics['closed_touch_present_fraction'] = observation.closed_touch_mask.any(-1).mean()
        return loss, metrics


@dataclasses.dataclass(frozen=True)
class ClosedLoopPiConfig(SingleStageLateTouchPiConfig):
    closed_block_size: int = 5
    closed_width: int = 64
    closed_limit: float = .1
    joint_objective: str = 'afce_single_stage_block_feedback_v1'

    def __post_init__(self):
        super().__post_init__()
        check_block_size(self.closed_block_size)

    def create(self, rng):
        return ClosedLoopPi(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = dataclasses.replace(obs,
                closed_state=jax.ShapeDtypeStruct((batch_size, 46), jnp.float32),
                closed_touch=jax.ShapeDtypeStruct((batch_size, 8, 54), jnp.float32),
                closed_touch_mask=jax.ShapeDtypeStruct((batch_size, 8), jnp.bool_),
                closed_offset=jax.ShapeDtypeStruct((batch_size,), jnp.int32))
        return obs, actions


@dataclasses.dataclass(frozen=True)
class ClosedLoopWeights:
    params_path: str

    def load(self, params):
        loaded = model_lib.restore_params(self.params_path, restore_type=np.ndarray)
        loaded = {k: v for k, v in loaded.items() if k not in ('action_in_proj', 'action_out_proj')}
        return weight_loaders._merge_params(loaded, params, missing_regex=(
            r'.*(lora|action_in_proj|action_out_proj|single_decoder|bimanual_decoder|effect_mean|effect_std|late_touch|closed_loop).*'))
