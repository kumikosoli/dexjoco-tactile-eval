"""Acknowledged H30 plan sessions; issue each block exactly once per plan.

Observation time is simulator/control time, not RPC arrival wall time. Clients
must acknowledge the executed offset and attach the tactile observation time.
No hidden action queue is allowed to silently replay a block after a timeout.
"""
from __future__ import annotations
import math
import threading
import numpy as np

from afce_cl.block import check_block_size
from afce_ss.late_touch import prepare_current_touch


class BlockSession:
    def __init__(self, make_plan, correct_block, *, action_dim, block_size=5):
        self.make_plan, self.correct_block = make_plan, correct_block
        self.action_dim = action_dim
        self.state_dim = 23*(action_dim//22)
        self.block_size = check_block_size(block_size)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.episode_id = None
        self.seen_episodes = set()
        self.plan_id = None
        self.used_ids = set()
        self.expected = 0
        self.plan = None
        self.anchor = None
        self.t0 = None

    def infer(self, observation):
        with self.lock:
            control = observation.get('closed_loop')
            if not isinstance(control, dict):
                raise ValueError('Missing explicit closed_loop session/offset/timestamps')
            plan_id = control.get('plan_id')
            offset = control.get('executed_steps')
            now, tactile_time = control.get('observation_time'), control.get('touch_time')
            if (not isinstance(plan_id, str) or not plan_id or isinstance(offset, bool)
                    or not isinstance(offset, int) or offset not in range(0, 30, self.block_size)
                    or not isinstance(now, (float, int)) or not isinstance(tactile_time, (float, int))
                    or not math.isfinite(now) or not math.isfinite(tactile_time) or abs(now-tactile_time) > 1e-5):
                raise ValueError('Invalid or stale feedback control envelope')
            state = np.asarray(observation['state'], np.float32)[:self.state_dim]
            if state.shape != (self.state_dim,) or not np.isfinite(state).all():
                raise ValueError('Current raw native state differs')
            touch, mask = prepare_current_touch(observation['touch'], observation['touch_mask'])
            new = bool(control.get('new_plan', False))
            episode = control.get('episode_id')
            if control.get('reset_session', False):
                if not new or offset != 0 or not isinstance(episode, str) or not episode or episode in self.seen_episodes:
                    raise ValueError('Explicit reset requires a fresh episode id and a new plan at offset zero')
                seen = self.seen_episodes | {episode}
                self.reset()
                self.seen_episodes, self.episode_id = seen, episode
            elif episode != self.episode_id:
                raise ValueError('Episode changed without an explicit session reset')
            if new:
                if offset != 0 or plan_id in self.used_ids or (self.plan_id is not None and self.expected != 30):
                    raise ValueError('New plan cannot overwrite an unfinished/reused plan')
                if self.t0 is not None and now < self.t0+30/50.-3e-5:
                    raise ValueError('New plan clock went backward before the completed horizon')
                raw = {k: v for k, v in observation.items() if k != 'closed_loop'}
                effect, plan = self.make_plan(raw)
                effect, plan = np.asarray(effect, np.float32), np.asarray(plan, np.float32)
                if effect.shape != (30, 256) or plan.shape != (30, self.action_dim):
                    raise ValueError('Plan must be complete E and native physical H30 actions')
                if not np.isfinite(effect).all() or not np.isfinite(plan).all():
                    raise ValueError('Nonfinite frozen/joint anchor plan')
                self.effect, self.plan, self.anchor = effect.copy(), plan.copy(), state.copy()
                self.plan_id, self.expected, self.t0 = plan_id, 0, float(now)
                self.used_ids.add(plan_id)
            if plan_id != self.plan_id or offset != self.expected or self.plan is None:
                raise ValueError('Missing/reordered/duplicate executed-block acknowledgement')
            if abs(float(now)-(self.t0+offset/50.)) > 3e-5:
                raise ValueError('Feedback time does not match acknowledged 50Hz execution')
            actions = np.asarray(self.correct_block(self.effect, self.plan, self.anchor, state,
                                                    touch, mask, offset), np.float32)
            if actions.shape != (self.block_size, self.action_dim) or not np.isfinite(actions).all():
                raise ValueError('Invalid next-block actions')
            self.expected += self.block_size
            return {'actions': actions, 'closed_loop': {'plan_id': self.plan_id,
                'start_offset': offset, 'end_offset': self.expected, 'horizon': 30,
                'touch_time': float(tactile_time), 'plan_complete': self.expected == 30}}
