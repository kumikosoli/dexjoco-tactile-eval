"""FTP-1's pretrained tactile expert as a JAX module beside the pi0.5 policy.

In FTP-1 the tactile tokens form their own block: they attend only to each other, and the action expert
attends to the image-language prefix and to the tactile block. The block's per-layer keys and values are
therefore a function of the tactile tokens alone, which FTP-1's own sampler uses (it caches them separately
and concatenates the caches). Here the expert is a separate one-expert Gemma stack that returns that cache.

Pretrained (FTP-1 v0426, 50k steps): the 18-layer expert (width 512, MLP 2048, 8 heads of 256, one key-value
head), the functional-area embedding (48 x 512) and the unified projection (LayerNorm, Linear, GELU, Linear).
Trained from scratch, as FTP-1 does for a sensor it has not seen: the tokenizer of this sensor's values, and
in the fusion scheme the encoder of each fingertip's recent history, whose output is added to that fingertip's
token and starts at zero."""
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as _gemma
from afce_touch.layout import AREAS, HISTORY_FRAMES

CONFIG = _gemma.Config(width=512, depth=18, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)


class HistoryBlock(nnx.Module):
    """Pre-norm self-attention over time and a two-layer MLP."""

    def __init__(self, width, heads, rngs):
        self.heads = heads
        self.attention_norm = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3*width, rngs=rngs)
        self.project = nnx.Linear(width, width, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(width, rngs=rngs)
        self.up = nnx.Linear(width, 2*width, rngs=rngs)
        self.down = nnx.Linear(2*width, width, rngs=rngs)

    def __call__(self, h):
        width = h.shape[-1]
        head = width//self.heads
        q, k, v = (z.reshape(*z.shape[:-1], self.heads, head).swapaxes(-2, -3)
                   for z in jnp.split(self.qkv(self.attention_norm(h)), 3, -1))
        attended = jax.nn.softmax(q@k.swapaxes(-1, -2)/jnp.sqrt(head), -1)@v
        h = h+self.project(attended.swapaxes(-2, -3).reshape(h.shape))
        return h+self.down(nnx.gelu(self.up(self.mlp_norm(h))))


class TouchHistory(nnx.Module):
    """One fingertip's last HISTORY_FRAMES summaries (force, normal and peak load, contact fraction, their
    change between frames, contact duration: 13 values per frame) -> a 512-wide vector for its token."""

    def __init__(self, rngs, width=128):
        self.embed = nnx.Linear(13, width, rngs=rngs)
        self.time = nnx.Param(jax.random.normal(rngs(), (HISTORY_FRAMES, width))*.02)
        self.first = HistoryBlock(width, 4, rngs)
        self.second = HistoryBlock(width, 4, rngs)
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.out = nnx.Linear(width, 512, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, frames):
        h = self.second(self.first(self.embed(frames)+self.time.value))
        return self.out(self.norm(h[..., -1, :]))


class TouchExpert(nnx.Module):
    def __init__(self, dtype, rngs, history=False):
        self.tokenizer_in = nnx.Linear(54, 512, rngs=rngs)
        self.tokenizer_out = nnx.Linear(512, 512, rngs=rngs)
        self.area_embedding = nnx.Param(jax.random.normal(rngs(), (48, 512))*.02)
        self.unified_norm = nnx.LayerNorm(512, epsilon=1e-5, rngs=rngs)
        self.unified_in = nnx.Linear(512, 512, rngs=rngs)
        self.unified_out = nnx.Linear(512, 512, rngs=rngs)
        self.expert = nnx_bridge.ToNNX(_gemma.Module(configs=[CONFIG], embed_dtype=dtype))
        # Called directly so that the unused text-embedding table is never created.
        self.expert.lazy_init([jnp.zeros((1, 1, 512))], jnp.zeros((1, 1), jnp.int32), jnp.zeros((1, 1, 1), bool), rngs=rngs)
        # Created last: everything above starts from the same values with and without the history encoder.
        if history:
            self.history = TouchHistory(rngs)

    def tokens(self, values, mask, history=None):
        """values: (B, 8, 54) standardized fingertip values; mask: (B, 8); history: (B, 8, HISTORY_FRAMES, 13)
        or None. FTP-1's order of operations."""
        keep = mask[..., None].astype(jnp.float32)
        x = self.tokenizer_out(nnx.gelu(self.tokenizer_in(values)))
        if history is not None:
            x = x+self.history(history)
        x = x*keep
        x = x+self.area_embedding.value[jnp.asarray(AREAS)]*keep
        # FTP-1's unified projection uses the exact GELU (torch.nn.GELU()).
        x = self.unified_out(nnx.gelu(self.unified_in(self.unified_norm(x)), approximate=False))
        return x*keep

    def cache(self, tokens, mask, positions):
        """Per-layer keys and values of the tactile block, attending within the block only."""
        _, cache = self.expert([tokens], positions, mask[:, :, None] & mask[:, None, :])
        return cache


def convert(state):
    """FTP-1 PyTorch tensors (numpy, by their safetensors names) -> the pure-dict parameters of TouchExpert,
    without the tokenizer. The mapping is openpi's own JAX<->PyTorch Gemma layout."""
    prefix = 'paligemma_with_expert.gemma_tactile_expert.model.'

    def layers(name):
        return np.stack([state[f'{prefix}layers.{i}.{name}'] for i in range(CONFIG.depth)]).astype(np.float32)
    heads, head, width = CONFIG.num_heads, CONFIG.head_dim, CONFIG.width
    q = layers('self_attn.q_proj.weight')            # (L, heads*head, width)
    k = layers('self_attn.k_proj.weight')            # (L, head, width)
    v = layers('self_attn.v_proj.weight')
    o = layers('self_attn.o_proj.weight')            # (L, width, heads*head)
    assert q.shape == (18, heads*head, width) and k.shape == (18, head, width) and o.shape == (18, width, heads*head)
    expert = {'layers': {
        'attn': {'q_einsum': {'w': q.reshape(18, heads, head, width).transpose(0, 1, 3, 2)},
                 'kv_einsum': {'w': np.stack([k.transpose(0, 2, 1), v.transpose(0, 2, 1)], 1)[:, :, None]},
                 'attn_vec_einsum': {'w': o.reshape(18, width, heads, head).transpose(0, 2, 3, 1)}},
        'mlp': {'gating_einsum': np.stack([layers('mlp.gate_proj.weight').transpose(0, 2, 1),
                                           layers('mlp.up_proj.weight').transpose(0, 2, 1)], 1),
                'linear': layers('mlp.down_proj.weight').transpose(0, 2, 1)},
        'pre_attention_norm': {'scale': layers('input_layernorm.weight')},
        'pre_ffw_norm': {'scale': layers('post_attention_layernorm.weight')}},
        'final_norm': {'scale': state[prefix+'norm.weight'].astype(np.float32)}}
    encoder = 'hpt_tactile_encoder.'
    return {'expert': expert,
            'area_embedding': state[encoder+'func_area_idx_embedding.weight'].astype(np.float32),
            'unified_norm': {'scale': state[encoder+'unified_proj.0.weight'].astype(np.float32),
                             'bias': state[encoder+'unified_proj.0.bias'].astype(np.float32)},
            'unified_in': {'kernel': state[encoder+'unified_proj.1.weight'].T.astype(np.float32),
                           'bias': state[encoder+'unified_proj.1.bias'].astype(np.float32)},
            'unified_out': {'kernel': state[encoder+'unified_proj.3.weight'].T.astype(np.float32),
                            'bias': state[encoder+'unified_proj.3.bias'].astype(np.float32)}}
