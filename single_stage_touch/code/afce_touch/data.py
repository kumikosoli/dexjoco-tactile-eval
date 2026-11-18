"""The tactile tokens of one dataset row: the observation before action[frame] and the history that led to
it, never a later row."""
import functools
import os
from pathlib import Path

import numpy as np

from afce_touch.layout import episode_summaries, packed_tokens


@functools.lru_cache(maxsize=128)
def _episode(root, task, episode):
    with np.load(Path(root)/task/f'episode_{episode:06d}.npz') as z:
        current = z['current']
    return current, episode_summaries(current)


def sample_touch(task, episode, frame):
    current, history = _episode(os.environ['AFCE_TOUCH_ROOT'], task, int(episode))
    if not 0 <= frame < len(current):
        raise IndexError(f'Missing touch at {task}/{episode}/{frame}')
    return packed_tokens(current[frame], history[:frame+1])
