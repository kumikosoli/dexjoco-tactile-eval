"""Adds the fingertip touch and its recent history to the official client's policy observation (fusion
scheme evaluation).

Importing this module patches DexJoCoOpenPIEnv: after start() the simulator's reset and step are wrapped with
touch_hook, the touch summary of every control interval since the reset is kept, and every processed
observation gains `touch` (8 x PACKED) and `touch_mask` (8), built by the same functions as the training
rows. With AFCE_TOUCH_OFF=1 the mask is all False: the policy then runs without touch (the no-touch control
of the same checkpoint). With AFCE_TOUCH_LOG set, each episode's per-interval touch summaries are saved there.
Actions, physics, rendering, termination and the control loop are untouched."""
import os
from pathlib import Path

import numpy as np

import dexjoco_openpi_client.dexjoco_openpi_env as client_env

from afce_touch.layout import features, packed_tokens, summaries
from touch_hook import TouchHook, stepped

Env = client_env.DexJoCoOpenPIEnv
_start, _close, _process_obs = Env.start, Env.close, Env._process_obs


def start(self):
    _start(self)
    env = self.env
    hook = TouchHook(env, self.env_name)
    reset, step = env.reset, env.step

    def record():
        row = features(hook.observation())
        self._touch_history.append(summaries(row, self._touch_history[-1][..., 6] if self._touch_history else None))

    def touch_reset(*args, **kwargs):
        save_history(self)
        out = reset(*args, **kwargs)
        hook.reset()
        self._touch_history = []
        record()
        return out

    def touch_step(action):
        out = stepped(hook, step, action)
        record()
        return out
    env.reset = touch_reset
    env.step = touch_step
    self._touch_hook = hook
    self._touch_history = []
    self._touch_episodes = 0
    print('TOUCH_HOOK_READY', self.env_name, hook.runtime.hand_count, hook.expected, flush=True)


def save_history(self):
    """One file per finished episode: (T+1, H, 4, 7) summaries, in the order the episodes ran."""
    root = os.environ.get('AFCE_TOUCH_LOG')
    if root and len(getattr(self, '_touch_history', [])) > 1:
        try:
            folder = Path(root)/self.env_name
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder/f'pid{os.getpid()}_episode_{self._touch_episodes:04d}.npy', np.stack(self._touch_history))
        except OSError as error:  # the log is a by-product; the evaluation does not depend on it
            print('TOUCH_LOG_NOT_SAVED', repr(error), flush=True)
        self._touch_episodes += 1


def close(self):
    hook = getattr(self, '_touch_hook', None)
    save_history(self)
    if hook is not None:
        hook.close()
        self._touch_hook = None
    _close(self)


def process_obs(self, env_obs):
    out = _process_obs(self, env_obs)
    row = features(self._touch_hook.observation())
    # Before the first reset there is no history: the current observation stands for all of it.
    history = np.stack(self._touch_history) if self._touch_history else summaries(row)[None]
    out['touch'], out['touch_mask'] = packed_tokens(row, history)
    if os.environ.get('AFCE_TOUCH_OFF') == '1':
        out['touch_mask'] = np.zeros_like(out['touch_mask'])
    return out


Env.start, Env.close, Env._process_obs = start, close, process_obs
