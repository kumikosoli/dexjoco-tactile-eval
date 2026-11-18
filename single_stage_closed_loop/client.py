"""Synchronous adapter: call only after executing the previous five actions.

The caller must obtain fresh state AND touch at simulator time `observation_time`;
it must not put this adapter behind the original H30 action queue.
"""
import uuid
import numpy as np


class BlockClient:
    def __init__(self, policy, block_size=5):
        if block_size != 5:
            raise ValueError('This client implements the deployed five-action protocol')
        self.policy, self.block_size = policy, block_size
        self.reset()

    def reset(self):
        self.episode = uuid.uuid4().hex
        self.executed = 0
        self.pending = False

    def infer(self, observation, *, observation_time, touch_time):
        if self.pending:
            raise ValueError('Acknowledge execution before requesting fresh feedback')
        offset = self.executed % 30
        payload = {**observation, 'closed_loop': {
            'episode_id': self.episode, 'reset_session': self.executed == 0,
            'plan_id': f'{self.episode}:{self.executed//30}', 'new_plan': offset == 0,
            'executed_steps': offset, 'observation_time': float(observation_time), 'touch_time': float(touch_time)}}
        result = self.policy.infer(payload)
        ack = result['closed_loop']
        actions = np.asarray(result['actions'])
        if (actions.shape[0] != self.block_size or ack['plan_id'] != payload['closed_loop']['plan_id']
                or ack['start_offset'] != offset or ack['end_offset'] != offset+self.block_size):
            raise ValueError('Server block acknowledgement differs; do not execute')
        self.pending = True
        return actions

    def acknowledge(self, executed_actions):
        if not self.pending or executed_actions != self.block_size:
            raise ValueError('Only acknowledge exactly the five actions actually executed')
        self.executed += self.block_size
        self.pending = False
