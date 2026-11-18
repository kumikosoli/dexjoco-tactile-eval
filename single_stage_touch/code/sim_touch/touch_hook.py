"""Fingertip touch for a simulator that does not publish it (official DexJoCo commit 8d23b0f).

The tactile package of DexJoCo commit 86f0197, the code that produced the replay data's touch, is used as is
(dexjoco_tactile_86f0197: byte copies of dexjoco/sim/tactile). That commit calls its runtime at four points of
an environment step; this module makes the same calls from outside the environment:

    reset()            -> runtime.reset()
    step(action) start -> runtime.begin()
    after each mujoco.mj_step on the environment's own data -> runtime.accumulate_substep()
    step(action) end   -> runtime.finish()

The runtime only reads contacts; it adds nothing to the model, so the trajectory is unchanged."""
import mujoco
import numpy as np

from dexjoco_tactile_86f0197 import (LEFT_FINGERTIP_BODY_NAMES, RIGHT_FINGERTIP_BODY_NAMES,
                                     SINGLE_FINGERTIP_BODY_NAMES, TactileRuntime)

_original_step = mujoco.mj_step
_active = []


def _step(model, data, *args, **kwargs):
    result = _original_step(model, data, *args, **kwargs)
    for hook in _active:
        if hook.stepping and data is hook.data:
            hook.runtime.accumulate_substep()
            hook.substeps += 1
    return result


mujoco.mj_step = _step


class TouchHook:
    def __init__(self, env, task):
        raw = env.unwrapped
        self.raw, self.model, self.data = raw, raw._model, raw._data
        names = ((RIGHT_FINGERTIP_BODY_NAMES, LEFT_FINGERTIP_BODY_NAMES) if task.startswith('bimanual_')
                 else (SINGLE_FINGERTIP_BODY_NAMES,))
        self.runtime = TactileRuntime(self.model, self.data, names, physics_dt=float(raw.physics_dt))
        self.expected = int(round(float(raw.control_dt)/float(raw.physics_dt)))
        self.stepping = False
        self.substeps = 0
        _active.append(self)

    def _same_simulation(self):
        if self.raw._model is not self.model or self.raw._data is not self.data:
            raise RuntimeError('The environment replaced its MuJoCo model or data; the touch runtime is stale')

    def reset(self):
        self._same_simulation()
        self.stepping = False
        self.runtime.reset()

    def begin(self):
        self._same_simulation()
        self.runtime.begin()
        self.substeps = 0
        self.stepping = True

    def finish(self):
        self.stepping = False
        if self.substeps != self.expected:
            raise RuntimeError(f'Expected {self.expected} physics steps in one control step, saw {self.substeps}')
        self.runtime.finish()

    def observation(self):
        """The tactile_gt dictionary of the last completed control interval (zeros after a reset)."""
        return self.runtime.observation()['tactile_gt']

    def close(self):
        if self in _active:
            _active.remove(self)


def stepped(hook, step, action):
    hook.begin()
    try:
        return step(action)
    finally:
        if hook.stepping:
            hook.finish()
