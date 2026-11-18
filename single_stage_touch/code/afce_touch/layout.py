"""Tactile tokens of scheme B: one per fingertip, in FTP-1's functional-area order (numpy only).

FTP-1 numbers the right hand's areas 0..23 and the left hand's 24..47; fingertips are 0 thumb, 1 index,
2 middle, 3 ring (4 little, absent on the Allegro hand). The replay stream orders fingertips index, middle,
ring, thumb and hands right, left; a single-arm task has only the right hand. Eight slots are always
returned (right then left, each thumb, index, middle, ring) with a mask for the hands that exist.

Each token carries the fingertip's 54 raw values: 48 taxel forces (16 x [tangent_u, tangent_v,
normal_inward]), Fx, Fy, Fz, normal load, peak load, contact fraction."""
import numpy as np

AREAS = np.array([0, 1, 2, 3, 24, 25, 26, 27], np.int32)
FINGER_TO_AREA_ORDER = (3, 0, 1, 2)  # thumb, index, middle, ring taken from index, middle, ring, thumb


def features(stream):
    """(H, 4, 54) from one tactile_gt observation (a dict of the arrays of the runtime or the zarr row)."""
    taxel = np.asarray(stream['taxel_force_local'], np.float32)
    hands = taxel.shape[0]
    assert taxel.shape == (hands, 4, 16, 3), taxel.shape
    return np.concatenate([taxel.reshape(hands, 4, 48), np.asarray(stream['fingertip_wrench_local'], np.float32)[..., :3],
                           np.asarray(stream['fingertip_normal'], np.float32)[..., None],
                           np.asarray(stream['fingertip_force_peak'], np.float32)[..., None],
                           np.asarray(stream['contact_fraction'], np.float32)[..., None]], -1)


def tokens(hand_features):
    """(8, 54) values and (8,) mask from (H, 4, 54) fingertip features, H = 1 (right) or 2 (right, left)."""
    hand_features = np.asarray(hand_features, np.float32)
    hands = hand_features.shape[0]
    assert hand_features.shape == (hands, 4, 54) and hands in (1, 2), hand_features.shape
    assert np.isfinite(hand_features).all()
    values = np.zeros((8, 54), np.float32)
    mask = np.zeros(8, bool)
    values[:4*hands] = hand_features[:, FINGER_TO_AREA_ORDER].reshape(4*hands, 54)
    mask[:4*hands] = True
    return values, mask


# Fusion scheme: besides the current values every token carries its fingertip's recent history.
HISTORY_FRAMES = 16
HISTORY_STRIDE = 3   # control steps; 16 frames cover the last 0.9 s at 50 Hz
PACKED = 54+HISTORY_FRAMES*7


def summaries(hand_features, run=None):
    """(H, 4, 7) from one (H, 4, 54) observation: Fx, Fy, Fz, normal load, peak load, contact fraction and the
    number of consecutive control intervals the fingertip has been in contact, given the previous row's count."""
    hand_features = np.asarray(hand_features, np.float32)
    contact = hand_features[..., 53] > 0
    run = contact.astype(np.float32) if run is None else (run+1)*contact
    return np.concatenate([hand_features[..., 48:54], run[..., None]], -1).astype(np.float32)


def episode_summaries(observations):
    """(T, H, 4, 7) from the (T, H, 4, 54) observations of one episode, in time order."""
    rows = []
    for row in observations:
        rows.append(summaries(row, rows[-1][..., 6] if rows else None))
    return np.stack(rows)


def packed_tokens(hand_features, history):
    """(8, PACKED) values and (8,) mask: the current fingertip values followed by HISTORY_FRAMES summaries,
    oldest first and ending with the current observation. `history` holds the summaries of every observation
    since the reset, (t+1, H, 4, 7); before the reset the first observation is repeated."""
    values, mask = tokens(hand_features)
    history = np.asarray(history, np.float32)
    hands = history.shape[1]
    last = len(history)-1
    index = np.maximum(last-HISTORY_STRIDE*np.arange(HISTORY_FRAMES-1, -1, -1), 0)
    window = history[index][:, :, FINGER_TO_AREA_ORDER].reshape(HISTORY_FRAMES, 4*hands, 7).transpose(1, 0, 2)
    packed = np.zeros((8, PACKED), np.float32)
    packed[:, :54] = values
    packed[:4*hands, 54:] = window.reshape(4*hands, HISTORY_FRAMES*7)
    return packed, mask
