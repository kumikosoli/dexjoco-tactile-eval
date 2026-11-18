"""Match the training TASK_IMAGE_KEYS duplication for native single-arm RPC inputs."""


def canonical_observation(observation, *, task):
    result = dict(observation)
    if task.startswith('bimanual_'):
        if not all(k in result for k in ('wrist_left', 'wrist_right')):
            raise ValueError('Bimanual observations need both real wrist images')
    elif 'wrist' in result:
        if 'wrist_left' in result or 'wrist_right' in result:
            raise ValueError('Ambiguous native and canonical wrist inputs')
        # The native single-arm dataset supplies the same wrist image in BOTH slots.
        result['wrist_left'] = result['wrist']
        result['wrist_right'] = result['wrist']
    elif not all(k in result for k in ('wrist_left', 'wrist_right')):
        raise ValueError('Missing native single-arm wrist image')
    return result
