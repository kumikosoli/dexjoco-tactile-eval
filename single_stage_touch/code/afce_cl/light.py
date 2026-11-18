"""Train only block correction from an immutable predicted-E cache; no pi optimizer."""
from __future__ import annotations
import argparse
import functools
import json
from pathlib import Path
import pickle
import signal
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from afce_cl.block import BlockCorrection, block_loss, take_block, check_block_size
from afce_cl.io import atomic, sha
from afce_cl.timeline import Timeline


class CorrectionPair(nnx.Module):
    def __init__(self, *, block_size=5, width=64, limit=.1, seed=42):
        rngs = nnx.Rngs(seed)
        self.single = BlockCorrection(22, block_size=block_size, width=width, limit=limit, rngs=rngs)
        self.bimanual = BlockCorrection(44, block_size=block_size, width=width, limit=limit, rngs=rngs)


class CacheDataset:
    def __init__(self, cache, data, touch, split, block_size=5, normalization_reference=None):
        self.cache = Path(cache)
        self.block_size = check_block_size(block_size)
        self.timeline = Timeline(data, touch)
        self.manifest = json.loads((self.cache/'manifest.json').read_text())
        identity = self.manifest['identity']
        self.normalization_reference = None
        if normalization_reference is not None:
            expected = identity.get('source_files', {}).get('decoder_init.npz')
            if expected is None or sha(normalization_reference) != expected:
                raise ValueError('Normalization reference differs from the frozen source identity')
            with np.load(normalization_reference, allow_pickle=False) as ref:
                self.normalization_reference = {k: ref[k] for k in ref.files}
        if (not self.manifest.get('complete') or identity['kind'] != 'actual_sampled_policy_E_and_frozen_late_plan_v1'
                or identity['source_updates'] != 60000 or identity['dataset_sha256'] != sha(Path(data)/'dataset_contract.json')
                or identity['touch_stats_sha256'] != sha(Path(touch)/'tactile_stats.json')):
            raise ValueError('Cache and admitted native data/source model differ')
        self.entries = [row for row in self.manifest['entries'] if row['split'] == split]
        self.samples = []
        # Native admitted replay plus predicted-plan cache fits in ~2 GB for 1097 episodes.
        # Retain per-episode arrays to avoid reopening 32 random parquet/npz files every update.
        self.loaded_episodes = {}
        for i, row in enumerate(self.entries):
            if sha(self.cache/row['file']) != row['sha256']:
                raise ValueError('Cache shard checksum mismatch')
            source = self.timeline.mapping[(row['task'], row['episode'])]
            if source['parquet_sha256'] != row['parquet_sha256'] or source['touch_sha256'] != row['touch_sha256']:
                raise ValueError('Cache episode mapping differs')
            for ai, frame in enumerate(row['anchors']):
                for offset in range(0, 30, block_size):
                    if frame+offset < source['frames']:
                        self.samples.append((i, ai, frame, offset))
        if not self.samples:
            raise ValueError(f'No correction samples in split={split}')

    def __len__(self):
        return len(self.samples)

    @functools.lru_cache(maxsize=2048)
    def shard(self, i):
        with np.load(self.cache/self.entries[i]['file'], allow_pickle=False) as f:
            result = {k: f[k] for k in f.files}
        for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            value = result[name]
            if value.dtype.kind == 'V':
                # NumPy's .npy format preserves bfloat16 bytes as V2, losing the dtype name.
                # Admit only the exact rounded, immutable source statistics, never guessed floats.
                if value.dtype.itemsize != 2 or self.normalization_reference is None:
                    raise ValueError('Legacy bfloat16 cache needs a SHA-bound source normalization reference')
                import ml_dtypes
                converted = value.view(ml_dtypes.bfloat16).astype(np.float32)
                dim = result['plan'].shape[-1]
                expected = self.normalization_reference[f'{name}_{dim}'].astype(ml_dtypes.bfloat16).astype(np.float32)
                if not np.array_equal(converted, expected):
                    raise ValueError('Legacy normalization bytes do not match the frozen source')
                result[name] = converted
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('Nonfinite cache shard')
        if np.any(result['state_std'] <= 0) or np.any(result['action_std'] <= 0):
            raise ValueError('Source normalization scale must be positive')
        return result

    def __getitem__(self, index):
        i, ai, frame, offset = self.samples[index]
        row, z = self.entries[i], self.shard(i)
        key = (row['task'], row['episode'])
        if key not in self.loaded_episodes:
            self.loaded_episodes[key] = self.timeline.episode(*key)
        states, actions, _, _ = self.loaded_episodes[key]
        # Reconfirm byte identities once when this episode is first consumed.
        self.verify_episode(row['task'], row['episode'], row['parquet_sha256'], row['touch_sha256'])
        dim, sd, n = actions.shape[1], states.shape[1], len(states)
        from afce_cl.timeline import aligned_feedback
        feedback = aligned_feedback(*[self.loaded_episodes[key][j] for j in (0, 2, 3)],
                                    frame, offset, block_size=self.block_size)
        valid = frame+offset+np.arange(self.block_size) < n
        targets = actions[np.minimum(frame+offset+np.arange(self.block_size), n-1)]
        anchor = (states[frame]-z['state_mean'])/z['state_std']
        current = (feedback['closed_state'][:sd]-z['state_mean'])/z['state_std']
        return {'effect': z['effect'][ai], 'plan': np.pad(z['plan'][ai], ((0, 0), (0, 44-dim))),
            'anchor': np.pad(anchor, (0, 46-sd)), 'current': np.pad(current, (0, 46-sd)),
            'touch': feedback['closed_touch'], 'mask': feedback['closed_touch_mask'],
            'offset': np.int32(offset), 'dim': np.int32(dim), 'valid': valid,
            'target': np.pad((targets-z['action_mean'])/z['action_std'], ((0, 0), (0, 44-dim)))}

    @functools.lru_cache(maxsize=None)
    def verify_episode(self, task, episode, parquet_sha, touch_sha):
        if (sha(self.timeline.data/task/f'data/chunk-000/file-{episode:03d}.parquet') != parquet_sha
                or sha(self.timeline.touch/task/f'episode_{episode:06d}.npz') != touch_sha):
            raise ValueError('Raw episode was mutated after cache generation')
        return True


def batch(dataset, rng, count):
    rows = [dataset[int(i)] for i in rng.integers(len(dataset), size=count)]
    return {k: jnp.asarray(np.stack([r[k] for r in rows])) for k in rows[0]}


def objective(model, b, scale):
    total = jnp.asarray(0.)
    action, finger = jnp.asarray(0.), jnp.asarray(0.)
    for dim, sd, module in [(22, 23, model.single), (44, 46, model.bimanual)]:
        planned = b['plan'][..., :dim]
        delta = module(b['effect'], planned, b['anchor'][:, :sd], b['current'][:, :sd],
                       b['touch'], b['mask'], b['offset'], scale)
        prediction = take_block(planned, b['offset'], module.block_size)+delta
        selected = b['dim'] == dim
        loss, metrics = block_loss(prediction, b['target'][..., :dim], b['valid'] & selected[:, None], dim)
        weight = selected.mean()
        total += weight*loss;action += weight*metrics['action_loss'];finger += weight*metrics['finger_delta_loss']
    return total, {'action_loss': action, 'finger_delta_loss': finger}


@nnx.jit
def train_step(optimizer, b, scale):
    (loss, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(optimizer.model, b, scale)
    norm = optax.global_norm(grads)
    optimizer.update(grads)
    return loss, {**metrics, 'grad_norm': norm}


@nnx.jit
def evaluate(model, b, scale):
    return objective(model, b, scale)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache', 'data', 'touch', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--block-size', type=int, default=5)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--limit', type=float, default=.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--stop-at-update', type=int, default=0)
    p.add_argument('--validation-batches', type=int, default=4)
    p.add_argument('--normalization-reference', type=Path,
                   help='Original source decoder_init.npz; required for legacy bfloat16 norm caches')
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.save_every, a.validation_batches) < 1 or a.lr <= 0:
        p.error('Positive budgets and learning rate required')
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.output/'training.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train = CacheDataset(a.cache, a.data, a.touch, 'train', a.block_size, a.normalization_reference)
    val = CacheDataset(a.cache, a.data, a.touch, 'val', a.block_size, a.normalization_reference)
    scale = jnp.asarray(json.loads((a.touch/'tactile_stats.json').read_text())['scale'], jnp.float32)
    if scale.shape != (54,) or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
        raise ValueError('Invalid immutable current-touch scale')
    config = {k: v for k, v in vars(a).items() if k not in ['stop_at_update', 'output']}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in config.items()}
    contract = {'schema': 'frozen_plan_block_correction_v1', 'config': config,
        'cache_manifest_sha256': sha(a.cache/'manifest.json'), 'source': train.manifest['identity'],
        'train_samples': len(train), 'val_samples': len(val),
        'validation_scope': 'correction holdout only; frozen policy trained on all1097',
        'block_code_sha256': sha(Path(__file__).with_name('block.py')), 'trainer_sha256': sha(__file__),
        'timeline_code_sha256': sha(Path(__file__).with_name('timeline.py')),
        'trainable': 'correction pair only; frozen backbone is absent from this optimizer'}
    path = a.output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Light-training resume contract changed')
    atomic(path, contract)
    model = CorrectionPair(block_size=a.block_size, width=a.width, limit=a.limit, seed=a.seed)
    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1), optax.adamw(a.lr, weight_decay=1e-4)))
    rng = np.random.default_rng(a.seed)
    checkpoint = a.output/'latest.pkl'
    step = 0
    if checkpoint.exists():
        receipt = json.loads((a.output/'latest.json').read_text())
        if sha(checkpoint) != receipt['sha256']:
            raise ValueError('Correction checkpoint checksum mismatch')
        with checkpoint.open('rb') as f: state = pickle.load(f)
        if state['contract'] != contract:
            raise ValueError('Saved correction source contract changed')
        nnx.update(optimizer, state['optimizer']);rng.bit_generator.state = state['rng'];step = state['step']
    stopped = []
    for sig in [signal.SIGTERM, signal.SIGUSR1]: signal.signal(sig, lambda s, f: stopped.append(s))
    begin = time.monotonic()

    def save():
        payload = {'contract': contract, 'optimizer': jax.device_get(nnx.state(optimizer)),
                   'rng': rng.bit_generator.state, 'step': step}
        temp = checkpoint.with_suffix('.pending')
        with temp.open('wb') as f: pickle.dump(payload, f, protocol=5)
        temp.replace(checkpoint)
        atomic(a.output/'latest.json', {'step': step, 'sha256': sha(checkpoint), 'complete': True})

    with (a.output/'metrics.jsonl').open('a', buffering=1) as log:
        log.write(json.dumps({'event': 'started', 'step': step, 'resumed': checkpoint.exists()})+'\n')
        while step < a.steps:
            before = time.monotonic()
            b = batch(train, rng, a.batch_size)
            loss, metrics = train_step(optimizer, b, scale)
            metrics = {k: float(v) for k, v in metrics.items()};metrics['loss'] = float(loss)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite light training metrics')
            step += 1
            if step <= 20 or step % 10 == 0:
                log.write(json.dumps({'event': 'train', 'step': step, 'utc': time.time(),
                    'step_seconds': time.monotonic()-before, **metrics})+'\n')
            if step % a.save_every == 0 or step == a.steps or stopped or (a.stop_at_update and step >= a.stop_at_update):
                save()
                vrng = np.random.default_rng(a.seed+100000)
                losses = [float(evaluate(model, batch(val, vrng, a.batch_size), scale)[0]) for _ in range(a.validation_batches)]
                log.write(json.dumps({'event': 'correction_holdout', 'step': step, 'loss': float(np.mean(losses))})+'\n')
            if step < a.steps and (stopped or (a.stop_at_update and step >= a.stop_at_update)):
                log.write(json.dumps({'event': 'stopped', 'step': step})+'\n')
                return 75
        atomic(a.output/'complete.json', {'step': step, 'complete': True, 'elapsed_seconds': time.monotonic()-begin,
                                          'performance_claimed': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
