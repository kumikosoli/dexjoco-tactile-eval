"""Single-stage AFCE data: every policy sample also carries the C01 encoder window (A, dS, V).

The encoder window is built by the unchanged C01 EvidenceData.sample code path (afce_all11.evidence),
restricted to one task and addressed by (episode, frame) instead of a global index.
Single-arm windows are zero-padded to the bimanual shapes; `effect_action_dim` tells them apart.
The effect target itself is filled in by the training loop from the EMA encoder (placeholder zeros here).
With AFCE_SS_TOUCH=1 every sample also carries scheme B's tactile tokens (afce_touch.data.sample_touch).
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from afce_all11.evidence import EvidenceData

ENC_KEYS = ('action', 'state', 'root', 'tips', 'joints', 'response', 'response_valid',
            'world', 'world_valid', 'world_query', 'world_target')
# bimanual shapes (single-arm windows are padded into these)
ENC_SHAPES = {'action': (30, 44), 'state': (46,), 'root': (30, 18), 'tips': (30, 8, 3), 'joints': (30, 32),
              'response': (30, 74), 'response_valid': (30,), 'world': (6, 8, 136), 'world_valid': (6, 8),
              'world_query': (6, 8, 4), 'world_target': (6, 8, 66)}


class TaskEvidence(EvidenceData):
    """EvidenceData for one task, all episodes, addressed by (episode, frame)."""

    def __init__(self, root, evidence, task):
        import pandas as pd
        root = Path(root)
        files = sorted((root/task/'data').rglob('*.parquet'))
        df = pd.concat([pd.read_parquet(f, columns=['episode_index', 'frame_index', 'action', 'observation.state']) for f in files])
        self.episodes, self.ends, total = [], [], 0
        self.by_episode = {}
        for ep, g in df.groupby('episode_index', sort=True):
            g = g.sort_values('frame_index')
            a = np.stack(g.action).astype(np.float32)
            s = np.stack(g['observation.state']).astype(np.float32)
            ad, sd = (44, 46) if task.startswith('bimanual_') else (22, 23)
            if a.shape != (len(g), ad) or s.shape != (len(g), sd):
                raise ValueError(f'{task}/{ep}: wrong dimensions')
            if not np.array_equal(g.frame_index.to_numpy(), np.arange(len(g))):
                raise ValueError('Noncontiguous frames')
            self.by_episode[int(ep)] = len(self.episodes)
            self.episodes.append({'task': task, 'episode': int(ep), 'actions': a, 'states': s})
            total += len(a)
            self.ends.append(total)
        self.ends = np.asarray(self.ends)
        self.total = total
        self.evidence = Path(evidence)
        manifest = json.loads((self.evidence/'world_complete.json').read_text())
        if not manifest.get('complete') or manifest.get('mask_mode') != 'legacy-none' or manifest.get('descriptor_dim') != 136:
            raise ValueError('Visual evidence must be the complete legacy-none C01 evidence')
        self.geo = {}
        self.world = {}

    def length(self, episode):
        return len(self.episodes[self.by_episode[episode]]['actions'])

    def window(self, episode, frame):
        ei = self.by_episode[episode]
        return self.sample((int(self.ends[ei-1]) if ei else 0) + frame)


def pad_window(row):
    out = {}
    for key in ENC_KEYS:
        value = np.asarray(row[key], dtype=np.float32)
        full = np.zeros(ENC_SHAPES[key], np.float32)
        full[tuple(slice(0, n) for n in value.shape)] = value
        out['enc_'+key] = full
    return out


class SingleStageDataset:
    """Replaces JointEffectCacheDataset: same joint-supervision fields plus the encoder window."""

    def __init__(self, dataset, root):
        self.dataset = dataset
        self.task = Path(root).name
        self.data_root = Path(os.environ['AFCE_SS_DATA'])
        self.evidence_root = Path(os.environ['AFCE_SS_EVIDENCE'])
        self.inputs = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['inputs'] = None
        return state

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.inputs is None:
            self.inputs = TaskEvidence(self.data_root, self.evidence_root, self.task)
        sample = dict(self.dataset[index])
        episode, frame = int(sample['episode_index']), int(sample['frame_index'])
        state = np.asarray(sample['observation.state'], dtype=np.float32)
        actions = np.asarray(sample['action'], dtype=np.float32)
        action_dim = actions.shape[-1]
        if action_dim not in (22, 44) or state.shape != (23*(action_dim//22),) or actions.shape != (30, action_dim):
            raise ValueError(f'Unexpected raw joint supervision: {state.shape}, {actions.shape}')
        window = self.inputs.window(episode, frame)
        # The encoder window and the policy sample must describe the same frame and the same future actions.
        if not np.array_equal(window['state'], state) or not np.allclose(window['action'], actions):
            raise ValueError(f'Encoder window does not match policy sample at {self.task}/{episode}/{frame}')
        sample['effect_target'] = np.zeros((30, 256), np.float32)
        sample['effect_state'] = np.pad(state, (0, 46-len(state)))
        sample['effect_actions'] = np.pad(actions, ((0, 0), (0, 44-action_dim)))
        sample['effect_action_dim'] = np.int32(action_dim)
        sample['effect_valid'] = frame + np.arange(30) < self.inputs.length(episode)
        sample.update(pad_window(window))
        if os.environ.get('AFCE_SS_TOUCH') == '1':
            # Scheme B's tactile tokens of this row (the observation before action[frame], never a later one).
            from afce_touch.data import sample_touch
            sample['touch'], sample['touch_mask'] = sample_touch(self.task, episode, frame)
        return sample


def unpad_groups(arrays, action_dims, device):
    """Turn padded enc_* numpy arrays (global batch order) into C01 grouped torch batches."""
    import torch
    groups = {}
    for d in (22, 44):
        idx = np.flatnonzero(action_dims == d)
        if not len(idx):
            continue
        h = d // 22
        cut = {'action': (slice(None), slice(0, d)), 'state': (slice(0, 23*h),), 'root': (slice(None), slice(0, 9*h)),
               'tips': (slice(None), slice(0, 4*h)), 'joints': (slice(None), slice(0, 16*h)),
               'response': (slice(None), slice(0, 37*h))}
        batch = {}
        for key in ENC_KEYS:
            value = arrays['enc_'+key][idx]
            if key in cut:
                value = value[(slice(None),)+cut[key]]
            batch[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        groups[d] = (idx, batch)
    return groups
