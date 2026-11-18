"""Single-stage AFCE policy (JAX): the C01 Stage-II model with a moving target.

Differences from afce_all11.single_finger_alignment.SingleFingerEffectPi:
  * effect_target is the EMA encoder's normalized E, written into the batch by the training loop;
  * flow matching is ramped in (fm_start .. fm_start+fm_ramp);
  * the policy decoders start from scratch, so they also learn to decode the clean target E
    (clean_decode_weight, all flow times, gradient only into the decoders);
  * the pooled prefix (current observation) feature is returned, stop-gradient, for the
    encoder-side contrastive loss;
  * rotation_loss='so3' (default 'rotvec' = unchanged) scores the rotation group of GT-AUX and clean
    decoding with the chordal SO(3) distance of the physical rotation vectors, as the encoder side does;
  * effect_prediction='x' (default 'velocity' = unchanged): the network output is the clean E and the velocity is
    (noisy - x)/max(t, X_PRED_MIN_T) (x-prediction with the velocity loss, Li & He 2025 "Back to Basics: Let Denoising
    Generative Models Denoise"); training loss, sampler and step count are otherwise unchanged;
  * time_sampling='uniform' (default 'beta' = pi0's Beta(1.5,1)): flow time t ~ U(.001, 1), the standard flow-matching
    choice, which spends more updates at low noise; the GT-AUX mid-time weight keeps its expected value;
  * time_sampling='shift': pi0's Beta(1.5,1) draw passed through the resolution-dependent timestep shift of
    Esser et al. 2024 (SD3), t' = s t/(1+(s-1)t) with s = sqrt(n/m): n = horizon x width of the flow target (30x256 E)
    and m = 50x32, the action chunk pi0's schedule was designed for. Higher-dimensional targets keep more signal at a
    given t, so more updates go to high noise. Training only: the Euler sampler is unchanged.
  * fm_time_weight='shift_mixture' (default 'none'): t keeps pi0's Beta(1.5,1) draw, and the flow-matching loss of
    each sample is multiplied by w(t) = 1 + p_shift(t)/p_beta(t), the density ratio of the SD3-shifted to the base
    distribution. Sampling density and loss weighting are interchangeable (Kingma & Gao 2023), so the effective
    flow-time density is the 50/50 mixture of 'beta' and 'shift'; unlike time_sampling='shift', GT-AUX, finger delta
    and the decoder see exactly the base distribution's samples. Training only.
  * gtaux_high=True (default False): GT-AUX and finger delta also act at t > 0.7, with the same per-sample weight as
    the [0.2, 0.7] gate, through stop-gradient copies of the decoders (the policy is supervised by the action error of
    its high-noise clean estimate; the decoders keep the [0.2, 0.7] supervision only).
  * with either switch (or policy_aux_scale != 1) the auxiliary term is split by gradient path: the decoders get v1's
    [0.2, 0.7] GT-AUX/finger-delta gradient unchanged, the policy gets policy_aux_scale times the gradient through
    stop-gradient decoder copies; policy_aux_scale is calibrated so that the policy-side output gradient ratio
    aux_to_fm_output_grad_ratio_unramped (afce_all11.single_finger_alignment) equals v1's (paper's calibration rule).
effect_mean/effect_std are fixed to 0/1: E is normalized on the encoder side.
"""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from afce_all11.pi_bridge_joint import JointEffectDataConfig
from afce_all11.readout_alignment import HUBER_DELTA, MID_PROBABILITY, grouped_loss
from afce_all11.single_finger_alignment import SingleFingerEffectPi, SingleFingerEffectPiConfig
from afce_ss.ss_data import ENC_KEYS
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask

X_PRED_MIN_T = .05
PI0_FLOW_SIZE = 50*32  # action chunk (horizon x width) pi0's Beta(1.5,1) schedule was designed for


def rotvec_matrix(vector):
    """Rodrigues, as afce_all11.geometry_objectives.rotvec_matrix; finite values and derivatives at the identity."""
    x, y, z = jnp.moveaxis(vector, -1, 0)
    zero = jnp.zeros_like(x)
    skew = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(vector.shape[:-1]+(3, 3))
    angle = jnp.sqrt(jnp.sum(jnp.square(vector), -1)+1e-12)
    a = jnp.sinc(angle/jnp.pi)[..., None, None]
    b = .5*jnp.square(jnp.sinc(angle/(2*jnp.pi)))[..., None, None]
    return jnp.eye(3, dtype=vector.dtype)+a*skew+b*(skew@skew)


def policy_action_loss(prediction, truth, valid, dimension, decoder, rotation):
    """grouped_loss (per-arm position / rotation / finger groups of normalized Huber). With rotation='so3' the
    rotation group is instead the chordal distance |R(p)-R(g)|_F^2/6 of the physical rotation vectors (the
    encoder's rotation_objective): rotation vectors are discontinuous at |theta|=pi, where equivalent
    orientations have distant components (Zhou et al., CVPR 2019; Hartley et al., IJCV 2013)."""
    if rotation == 'rotvec':
        return grouped_loss(prediction, truth, valid, dimension)
    if rotation != 'so3':
        raise ValueError(f'Unknown rotation loss {rotation}')
    mean, std = jax.lax.stop_gradient(decoder.action_mean.value), jax.lax.stop_gradient(decoder.action_std.value)
    count = jnp.maximum(valid.sum(1), 1)
    groups = []
    for offset in range(0, dimension, 22):
        for lo, hi in ((0, 3), (3, 6), (6, 22)):
            a, b = offset+lo, offset+hi
            if lo == 3:
                physical = lambda x: x[..., a:b]*std[a:b]+mean[a:b]
                values = jnp.sum(jnp.square(rotvec_matrix(physical(prediction))-rotvec_matrix(physical(truth))), (-1, -2))/6
                groups.append(jnp.sum(jnp.where(valid, values, 0), 1)/count)
                continue
            error = prediction[..., a:b]-truth[..., a:b]
            absolute = jnp.abs(error)
            values = jnp.where(absolute < HUBER_DELTA, .5*error**2, absolute-.5)
            weights = jnp.broadcast_to(valid[..., None], values.shape)
            groups.append(jnp.sum(jnp.where(weights, values, 0), (1, 2))/jnp.maximum(weights.sum((1, 2)), 1))
    return jnp.stack(groups).mean(0)


class SingleStagePi(SingleFingerEffectPi):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.fm_start = config.fm_start
        self.fm_ramp = config.fm_ramp
        self.clean_decode_weight = config.clean_decode_weight
        self.rotation_loss = config.rotation_loss
        self.effect_prediction = config.effect_prediction
        self.time_sampling = config.time_sampling
        if self.effect_prediction not in ('velocity', 'x') or self.time_sampling not in ('beta', 'uniform', 'shift'):
            raise ValueError(f'Unknown effect_prediction/time_sampling {self.effect_prediction}/{self.time_sampling}')
        self.time_shift = float(np.sqrt(self.action_horizon*self.action_dim/PI0_FLOW_SIZE))  # 30x256 E: 2.19
        self.fm_time_weight = config.fm_time_weight
        self.gtaux_high = config.gtaux_high
        self.policy_aux_scale = config.policy_aux_scale
        self.decoupled_aux = self.gtaux_high or self.fm_time_weight != 'none' or self.policy_aux_scale != 1.0
        if self.fm_time_weight not in ('none', 'shift_mixture'):
            raise ValueError(f'Unknown fm_time_weight {self.fm_time_weight}')
        if (self.fm_time_weight != 'none' or self.gtaux_high) and (self.time_sampling != 'beta' or self.effect_prediction != 'velocity'):
            raise ValueError('fm_time_weight / gtaux_high are defined on top of the base (beta, velocity) model')

    def shift_density_ratio(self, time):
        """p_shift(t)/p_beta(t) for t = .999 u + .001, u ~ Beta(1.5, 1), and its SD3 shift t' = s t/(1+(s-1)t)."""
        s = self.time_shift
        base = time/(s-(s-1)*time)                     # pre-image of t under the shift
        jacobian = s/jnp.square(s-(s-1)*time)          # d base / d t
        u_base = jnp.clip((base-.001)/.999, 0., 1.)
        u_time = jnp.clip((time-.001)/.999, 1e-12, 1.)
        return jnp.where(base >= .001, jnp.sqrt(u_base/u_time)*jacobian, 0.)

    def fm_sample_weight(self, time):
        if self.fm_time_weight == 'shift_mixture':
            return jax.lax.stop_gradient(1.+self.shift_density_ratio(time))
        return jnp.ones_like(time)

    def high_time_weight(self, time):
        """t > 0.7 with the [0.2, 0.7] gate's per-sample weight (same normalizer as mid_time_weight)."""
        return (time > .7).astype(jnp.float32)*self.mid_time_weight(jnp.full_like(time, .45))

    @staticmethod
    def frozen(decoder):
        graphdef, state = nnx.split(decoder)
        return nnx.merge(graphdef, jax.tree.map(jax.lax.stop_gradient, state))

    def frozen_aux_terms(self, estimate, observation, weights):
        """GT-AUX and finger delta (as auxiliary / finger_delta_loss) through stop-gradient decoder copies."""
        physical = estimate*self.effect_std.value+self.effect_mean.value
        gt_losses, finger_losses = [], []
        for decoder, dim, state_dim in ((self.single_decoder, 22, 23), (self.bimanual_decoder, 44, 46)):
            decoder = self.frozen(decoder)
            prediction = decoder(physical, observation.effect_state[:, :state_dim])
            truth = jax.lax.stop_gradient((observation.effect_actions[..., :dim]-decoder.action_mean.value)/decoder.action_std.value)
            gt_losses.append(policy_action_loss(prediction, truth, observation.effect_valid, dim, decoder, self.rotation_loss))
            slices = ((6, 22),) if dim == 22 else ((6, 22), (28, 44))
            pf = jnp.concatenate([prediction[..., lo:hi] for lo, hi in slices], -1)
            tf = jnp.concatenate([truth[..., lo:hi] for lo, hi in slices], -1)
            pd, td = pf[:, 1:]-pf[:, :-1], tf[:, 1:]-tf[:, :-1]
            valid = observation.effect_valid[:, 1:] & observation.effect_valid[:, :-1] & (observation.effect_action_dim == dim)[:, None]
            mask = jnp.broadcast_to(valid[..., None], td.shape)
            scale = jax.lax.stop_gradient(jnp.maximum(jnp.sqrt(
                jnp.sum(jnp.where(mask, jnp.square(td), 0), axis=(0, 1), keepdims=True)
                / jnp.maximum(mask.sum(axis=(0, 1), keepdims=True), 1)+1e-8), self.finger_delta_scale_floor))
            error = (pd-td)/scale
            absolute = jnp.abs(error)
            huber = jnp.where(absolute < 1., .5*jnp.square(error), absolute-.5)
            finger_losses.append(jnp.sum(jnp.where(mask, huber, 0), axis=(1, 2))/jnp.maximum(mask.sum(axis=(1, 2)), 1))
        single = observation.effect_action_dim == 22
        gt = jnp.where(single, gt_losses[0], gt_losses[1])
        finger = jnp.where(single, finger_losses[0], finger_losses[1])
        return jnp.mean(weights*gt), jnp.mean(weights*finger)

    def shift_time(self, time):
        return self.time_shift*time/(1+(self.time_shift-1)*time)

    def sample_time(self, rng, shape):
        if self.time_sampling == 'uniform':
            return jax.random.uniform(rng, shape)*.999+.001
        time = jax.random.beta(rng, 1.5, 1, shape)*.999+.001
        return self.shift_time(time) if self.time_sampling == 'shift' else time

    def mid_time_weight(self, time):
        """GT-AUX gate I[.2<=t<=.7] divided by its probability under the flow-time distribution."""
        if self.time_sampling == 'uniform':
            probability = .5/.999
        elif self.time_sampling == 'shift':
            inverse = lambda t: t/(self.time_shift-(self.time_shift-1)*t)  # base time mapped to t
            cdf = lambda t: ((inverse(t)-.001)/.999)**1.5
            probability = cdf(.7)-cdf(.2)
        else:
            probability = MID_PROBABILITY
        return ((time >= .2) & (time <= .7)).astype(jnp.float32)/probability

    def velocity_from_output(self, output, noisy, time):
        if self.effect_prediction == 'x':
            return (noisy-output)/jnp.maximum(time, X_PRED_MIN_T)
        return output

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        """openpi Pi0.sample_actions (Euler, t=1 noise -> t=0) with the configurable output parameterization."""
        if self.effect_prediction == 'velocity':
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
        import einops
        observation = model_lib.preprocess_observation(None, observation, train=False)
        dt = -1.0/num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1)-1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar_mask), positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=-1)-1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions,
                                                             kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
            output = self.action_out_proj(suffix_out[:, -self.action_horizon:]).astype(jnp.float32)
            v_t = self.velocity_from_output(output, x_t, time)
            return x_t+dt*v_t, time+dt

        def cond(carry):
            return carry[1] >= -dt/2
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def auxiliary(self, estimate, reference, observation, weights, mode):
        """SingleFingerEffectPi.auxiliary (GT-AUX) with the configurable rotation group."""
        if mode != 'ground_truth':
            raise ValueError('Trainable GT-AUX only supports the ground-truth action target')
        del reference
        physical = estimate*self.effect_std.value+self.effect_mean.value
        losses = []
        for decoder, dim, state_dim in ((self.single_decoder, 22, 23), (self.bimanual_decoder, 44, 46)):
            prediction = decoder(physical, observation.effect_state[:, :state_dim])
            truth = jax.lax.stop_gradient((observation.effect_actions[..., :dim]-decoder.action_mean.value)/decoder.action_std.value)
            losses.append(policy_action_loss(prediction, truth, observation.effect_valid, dim, decoder, self.rotation_loss))
        single = observation.effect_action_dim == 22
        per_sample = jnp.where(single, losses[0], losses[1])
        return jnp.mean(weights*per_sample), (per_sample, single)

    def flow_with_prefix(self, rng, observation, train):
        target = jax.lax.stop_gradient(observation.effect_target)
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(pre_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, target.shape)
        time = self.sample_time(time_rng, target.shape[:-2])
        t = time[..., None, None]
        noisy = t*noise+(1-t)*target
        truth_velocity = noise-target
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, noisy, time)
        mask = jnp.concatenate([prefix_mask, suffix_mask], 1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], 0)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix, suffix], mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1)-1, adarms_cond=[None, condition])
        velocity = self.velocity_from_output(self.action_out_proj(suffix_out[:, -self.action_horizon:]).astype(jnp.float32), noisy, t)
        if self.fm_time_weight == 'none':
            fm = jnp.mean(jnp.square(velocity-truth_velocity))
        else:
            fm = jnp.mean(jnp.square(velocity-truth_velocity)*self.fm_sample_weight(time)[..., None, None])
        weights = prefix_mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(prefix_out.astype(jnp.float32)*weights, 1)/jnp.maximum(weights.sum(1), 1)
        return fm, noisy-t*velocity, target, velocity, truth_velocity, time, jax.lax.stop_gradient(pooled)

    def clean_decode_loss(self, observation):
        target = jax.lax.stop_gradient(observation.effect_target)
        losses = []
        for decoder, dim, state_dim in ((self.single_decoder, 22, 23), (self.bimanual_decoder, 44, 46)):
            prediction = decoder(target, observation.effect_state[:, :state_dim])
            truth = jax.lax.stop_gradient((observation.effect_actions[..., :dim]-decoder.action_mean.value)/decoder.action_std.value)
            losses.append(policy_action_loss(prediction, truth, observation.effect_valid, dim, decoder, self.rotation_loss))
        single = observation.effect_action_dim == 22
        return jnp.mean(jnp.where(single, losses[0], losses[1]))

    def compute_loss_and_metrics_at_step(self, rng, observation, actions, *, step, train=False):
        if observation.effect_valid is None:
            raise ValueError('Aligned valid frame masks are required')
        fm, clean, target, velocity, truth, time, pooled = self.flow_with_prefix(rng, observation, train)

        def auxiliary_objective(value):
            gtaux, (per_sample, single) = self.auxiliary(value, target, observation, self.mid_time_weight(time), 'ground_truth')
            finger, _ = self.finger_delta_loss(value, observation, self.mid_time_weight(time))
            if self.gtaux_high:
                gt_high, finger_high = self.frozen_aux_terms(value, observation, self.high_time_weight(time))
                gtaux, finger = gtaux+gt_high, finger+finger_high
            return self.alignment_weight*gtaux+self.finger_delta_weight*finger, (gtaux, finger, per_sample, single)
        raw, (gtaux, finger, per_sample, single) = auxiliary_objective(clean)
        if self.decoupled_aux:
            # Same gradients as the joint term at policy_aux_scale = 1, split by path: the decoders receive v1's
            # [0.2, 0.7] GT-AUX/finger-delta gradient (policy estimate stop-gradient); the policy receives the
            # (calibrated) policy_aux_scale times the gradient through stop-gradient decoder copies.
            weights = self.mid_time_weight(time)+(self.high_time_weight(time) if self.gtaux_high else 0.)
            gt_p, finger_p = self.frozen_aux_terms(clean, observation, weights)
            policy_raw = self.policy_aux_scale*(self.alignment_weight*gt_p+self.finger_delta_weight*finger_p)
            fixed = jax.lax.stop_gradient(clean)
            gt_d, _ = self.auxiliary(fixed, target, observation, self.mid_time_weight(time), 'ground_truth')
            finger_d, _ = self.finger_delta_loss(fixed, observation, self.mid_time_weight(time))
            raw = policy_raw+self.alignment_weight*gt_d+self.finger_delta_weight*finger_d
        # Output-space gradient ratio |d aux / d velocity| / |d FM / d velocity| (afce_all11.single_finger_alignment),
        # stop-gradient diagnostic for the paper's calibration rule (~0.1 weighted median).
        aux_grad = jax.grad(lambda v: auxiliary_objective(v)[0])(jax.lax.stop_gradient(clean))
        fm_grad = 2*(velocity-truth)*self.fm_sample_weight(time)[..., None, None]/velocity.size
        ratio = jax.lax.stop_gradient(jnp.linalg.norm(time[..., None, None]*aux_grad)/jnp.maximum(jnp.linalg.norm(fm_grad), 1e-12))
        ramp = jnp.clip((step-self.alignment_start)/self.alignment_ramp_steps, 0., 1.)
        fm_weight = jnp.clip((step-self.fm_start)/self.fm_ramp, 0., 1.)
        clean_decode = self.clean_decode_loss(observation)
        total = fm_weight*fm+ramp*raw+self.clean_decode_weight*clean_decode
        metrics = {
            'loss': total, 'effect_loss': fm, 'fm_weight': fm_weight, 'action_loss': ramp*raw,
            'gtaux_loss': self.alignment_weight*ramp*gtaux, 'finger_delta_loss': self.finger_delta_weight*ramp*finger,
            'gtaux_loss_raw': gtaux, 'finger_delta_loss_raw': finger, 'ramp': ramp,
            'clean_decode_loss': clean_decode, 'action_loss_raw': per_sample.mean(),
            'single_samples': single.sum().astype(jnp.float32), 'bimanual_samples': (~single).sum().astype(jnp.float32),
            'valid_fraction': observation.effect_valid.mean(), 'time_mean': time.mean(),
            'target_rms': jnp.sqrt(jnp.mean(jnp.square(target))),
            'aux_to_fm_output_grad_ratio_unramped': ratio, 'policy_aux_to_fm_ratio': ratio*self.policy_aux_scale,
            'obs_feature': pooled,
        }
        return total, metrics


@dataclasses.dataclass(frozen=True)
class SingleStagePiConfig(SingleFingerEffectPiConfig):
    joint_objective: str = 'afce_single_stage_c01_contrastive_v1'
    fm_start: int = 5000
    fm_ramp: int = 5000
    clean_decode_weight: float = 1.0
    rotation_loss: str = 'rotvec'
    effect_prediction: str = 'velocity'
    time_sampling: str = 'beta'
    fm_time_weight: str = 'none'
    gtaux_high: bool = False
    policy_aux_scale: float = 1.0

    def create(self, rng):
        return SingleStagePi(self, nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class PreserveTargets(transforms.DataTransformFn):
    inputs: tuple
    names: tuple

    def __call__(self, data):
        extra = {name: data[name] for name in self.names if name in data}
        for transform in self.inputs:
            data = transform(data)
        return {**data, **extra}


SS_FIELDS = ('effect_state', 'effect_actions', 'effect_action_dim', 'effect_valid') + tuple('enc_'+k for k in ENC_KEYS)


@dataclasses.dataclass(frozen=True)
class SingleStageDataConfig(JointEffectDataConfig):
    touch: bool = False  # also carry scheme B's tactile tokens (needs scheme B's patched code copy)

    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        repack = base.repack_transforms.inputs[0]
        names = SS_FIELDS+(('touch', 'touch_mask') if self.touch else ())
        fields = {name: name for name in names}
        inner = base.data_transforms.inputs[0].inputs  # unwrap PreserveJointTargets
        return dataclasses.replace(base,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({**repack.structure, **fields})]),
            data_transforms=dataclasses.replace(base.data_transforms, inputs=[PreserveTargets(tuple(inner), names)]))
