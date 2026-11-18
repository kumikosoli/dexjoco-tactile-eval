"""Versioned native replay alignment; never join episodes by a global frame index."""
from __future__ import annotations
import functools
import json
from pathlib import Path
import numpy as np

from afce_touch.layout import tokens
from afce_cl.block import check_block_size


def aligned_feedback(states, timestamps, current_touch, anchor, offset, *, block_size=5):
    check_block_size(block_size)
    n = len(states)
    if not 0 <= anchor < n or offset not in range(0, 30, block_size) or anchor+offset >= n:
        raise ValueError('Feedback offset outside this episode or H30')
    if timestamps.shape != (n,) or len(current_touch) != n:
        raise ValueError('Touch/state/clock episode lengths differ')
    if not np.allclose(timestamps, timestamps[0]+np.arange(n)/50., atol=3e-5, rtol=0):
        raise ValueError('Native replay must have real contiguous 50Hz timestamps')
    frame = anchor+offset
    touch, mask = tokens(current_touch[frame])
    # Crucially, no history/current_touch[frame+1] is requested here.
    return {'closed_state': np.pad(states[frame], (0, 46-states.shape[1])).astype(np.float32),
            'closed_touch': touch, 'closed_touch_mask': mask, 'closed_offset': np.int32(offset)}


class Timeline:
    def __init__(self, data, touch):
        self.data, self.touch = Path(data), Path(touch)
        contract = json.loads((self.data/'dataset_contract.json').read_text())
        if not contract.get('complete'):
            raise ValueError('Only admitted reconstructed replay is supported')
        self.mapping = {(r['task'], int(r['episode'])): r for r in contract['mapping']}

    @functools.lru_cache(maxsize=8)
    def episode(self, task, episode):
        import pandas as pd
        row = self.mapping[(task, int(episode))]
        path = self.data/task/f'data/chunk-000/file-{episode:03d}.parquet'
        df = pd.read_parquet(path, columns=['episode_index', 'frame_index', 'timestamp', 'action', 'observation.state'])
        if not (df.episode_index == episode).all() or not np.array_equal(df.frame_index, np.arange(len(df))):
            raise ValueError('Noncontiguous/ambiguous episode identity')
        states = np.stack(df['observation.state']).astype(np.float32)
        actions = np.stack(df.action).astype(np.float32)
        timestamps = df.timestamp.to_numpy(np.float64)
        dim = 44 if task.startswith('bimanual_') else 22
        with np.load(self.touch/task/f'episode_{episode:06d}.npz', allow_pickle=False) as z:
            touch = z['current']
        if states.shape != (row['frames'], 23*(dim//22)) or actions.shape != (len(states), dim):
            raise ValueError('Native action/state dimension changed')
        aligned_feedback(states, timestamps, touch, 0, 0)
        if not all(np.isfinite(x).all() for x in [states, actions, touch, timestamps]):
            raise ValueError('Nonfinite replay values')
        return states, actions, timestamps, touch

    def feedback(self, task, episode, anchor, offset, block_size):
        states, _, time, touch = self.episode(task, episode)
        return aligned_feedback(states, time, touch, anchor, offset, block_size=block_size)
