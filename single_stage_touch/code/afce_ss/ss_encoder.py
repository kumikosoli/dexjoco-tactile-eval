"""Torch side of single-stage AFCE: the unchanged C01 Query encoder trained online, an EMA copy that
produces the policy target, running E normalization, and a SigLIP-style contrastive loss between
the encoder's E and the policy's (stop-gradient) current-observation feature.

C01 losses are exactly those of afce_all11.train_c01 (grouped action Huber, lambda_S/V ramp
.05->.2 over 1000 steps, position term /3, rotation .05).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from afce_all11.codec import losses
from afce_all11.geometry_objectives import rotation_objective
from afce_all11.tuned_codec import TunedCodec


class Heads(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        self.o = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, 512), nn.GELU(), nn.Linear(512, 256))
        self.log_t = nn.Parameter(torch.tensor(math.log(10.)))
        self.bias = nn.Parameter(torch.tensor(-10.))

    def forward(self, e, o):
        ze = F.normalize(self.e(e.mean(1)), dim=-1)
        zo = F.normalize(self.o(o), dim=-1)
        logits = ze @ zo.T * self.log_t.exp() + self.bias
        labels = 2*torch.eye(len(e), device=e.device)-1
        loss = -F.logsigmoid(labels*logits).sum()/len(e)
        accuracy = (logits.argmax(1) == torch.arange(len(e), device=e.device)).float().mean()
        return loss, accuracy


class SingleStageEncoder:
    def __init__(self, stats, device, obs_dim=2048, nce_weight=.1, ema_decay=.999, norm_decay=.999,
                 lr=1e-4, lr_low=3e-5, lr_switch=40000, std_floor=1e-3, seed=42):
        torch.manual_seed(seed)
        self.stats, self.device = stats, device
        self.model = TunedCodec('query', 'ASV', stats, False).to(device)
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.heads = Heads(obs_dim).to(device)
        self.opt = torch.optim.AdamW(list(self.model.parameters())+list(self.heads.parameters()),
                                     lr=lr, betas=(.9, .99), eps=1e-8, weight_decay=.01)
        self.nce_weight, self.ema_decay, self.norm_decay = nce_weight, ema_decay, norm_decay
        self.lr, self.lr_low, self.lr_switch, self.std_floor = lr, lr_low, lr_switch, std_floor
        self.mean = None
        self.var = None

    # ---- target -------------------------------------------------------------------------------
    @torch.no_grad()
    def target(self, groups, batch_size):
        e = torch.zeros(batch_size, 30, 256, device=self.device)
        for d, (idx, b) in groups.items():
            e[torch.as_tensor(idx, device=self.device)] = self.ema.encode(b).float()
        flat = e.reshape(-1, 256)
        m, v = flat.mean(0), flat.var(0, unbiased=False)
        if self.mean is None:
            self.mean, self.var = m, v
        else:
            self.mean = self.norm_decay*self.mean+(1-self.norm_decay)*m
            self.var = self.norm_decay*self.var+(1-self.norm_decay)*v
        std = self.var.sqrt().clamp_min(self.std_floor)
        return ((e-self.mean)/std).cpu().numpy().astype(np.float32)

    # ---- online update -------------------------------------------------------------------------
    def update(self, groups, batch_size, obs_feature, step):
        self.model.train()
        lam = .05+.15*min(step/1000, 1.)
        self.opt.zero_grad(set_to_none=True)
        values = {}
        e_all = torch.zeros(batch_size, 30, 256, device=self.device)
        total = 0.
        for d, (idx, b) in groups.items():
            out = self.model(b)
            v = losses(out, b, self.stats, lam, lam)
            scale = self.stats[str(d)]['action_std'].to(self.device)
            pos = torch.stack([F.smooth_l1_loss(out['action'][..., o:o+3]/scale[o:o+3], b['action'][..., o:o+3]/scale[o:o+3])
                               for o in range(0, d, 22)]).mean()
            rotation = rotation_objective(out['action'], b['action'])
            v['loss'] = v['loss']+pos/3+.05*rotation
            v['rotation'] = rotation
            weight = len(idx)/batch_size
            total = total+v['loss']*weight
            for k, x in v.items():
                values['enc_'+k] = values.get('enc_'+k, 0.)+float(x.detach())*weight
            e_all = e_all.index_copy(0, torch.as_tensor(idx, device=self.device), out['E'].float())
        obs = torch.as_tensor(obs_feature, device=self.device, dtype=torch.float32)
        nce, accuracy = self.heads(e_all, obs)
        total = total+self.nce_weight*nce
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite encoder loss')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(self.model.parameters())+list(self.heads.parameters()), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite encoder gradient')
        for group in self.opt.param_groups:
            group['lr'] = self.lr if step <= self.lr_switch else self.lr_low
        self.opt.step()
        with torch.no_grad():
            for p, q in zip(self.model.parameters(), self.ema.parameters()):
                q.mul_(self.ema_decay).add_(p.detach(), alpha=1-self.ema_decay)
            for p, q in zip(self.model.buffers(), self.ema.buffers()):
                q.copy_(p)
            e_std = e_all.reshape(-1, 256).std(0).mean()
        values.update(enc_nce=float(nce.detach()), enc_nce_accuracy=float(accuracy), enc_grad_norm=float(grad),
                      enc_total=float(total.detach()), enc_E_channel_std=float(e_std),
                      enc_temperature=float(self.heads.log_t.exp()), enc_norm_std_mean=float(self.var.sqrt().mean()))
        return values

    # ---- checkpoint ----------------------------------------------------------------------------
    def state_dict(self):
        return {'model': self.model.state_dict(), 'ema': self.ema.state_dict(), 'heads': self.heads.state_dict(),
                'opt': self.opt.state_dict(), 'mean': self.mean, 'var': self.var,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(self.device)}

    def load_state_dict(self, blob):
        self.model.load_state_dict(blob['model'])
        self.ema.load_state_dict(blob['ema'])
        self.heads.load_state_dict(blob['heads'])
        self.opt.load_state_dict(blob['opt'])
        self.mean = None if blob['mean'] is None else blob['mean'].to(self.device)
        self.var = None if blob['var'] is None else blob['var'].to(self.device)
        torch.set_rng_state(blob['torch_rng'])
        torch.cuda.set_rng_state(blob['cuda_rng'], self.device)

    def decoder_arrays(self):
        """C01 decoder arrays in the export_joint_decoder layout (for the JAX policy decoders' init)."""
        arrays = {}
        sd = self.model.state_dict()
        for d in (22, 44):
            prefix = f'action_decoders.{d}.'
            for name, value in sd.items():
                if name.startswith(prefix):
                    arrays[f'decoder_{d}/'+name[len(prefix):]] = value.detach().cpu().numpy()
            for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
                arrays[f'{name}_{d}'] = sd[f'{name}_{d}'].detach().cpu().numpy()
        arrays['effect_mean'] = np.zeros(256, np.float32)
        arrays['effect_std'] = np.ones(256, np.float32)
        return arrays
