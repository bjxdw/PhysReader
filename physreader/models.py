"""Models predicting z = (log10 E/rho, log10 sigma_Y/rho).

Teacher    exact 3D surface deformation (simulation only)
PhysReader video + 2D tracklets; regresses the teacher's 288-d latent, decoded by the frozen teacher head
Predictor  direct regression from a chosen input combination (the other rows of Table 1)
"""
import math

import torch
from torch import nn

from .data import CHANNELS
from .nets import SurfaceBlock, TrackEncoder, surface_features
from .vjepa import load_vjepa2, video_features

LATENT = 288


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(LATENT, 128), nn.LayerNorm(128), nn.GELU())
        self.E = nn.Linear(128, 1)
        self.YS = nn.Linear(128, 1)
        nn.init.constant_(self.E.bias, 3.5)
        nn.init.constant_(self.YS.bias, 3.1505)

    def forward(self, z):
        h = self.head(z)
        return torch.cat([self.E(h), self.YS(h)], -1).float()


class Teacher(nn.Module):
    """Graph network on 1024 exact surface particles over the 16-frame window."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(10, 128), nn.LayerNorm(128), nn.GELU())
        self.blocks = nn.ModuleList([SurfaceBlock(d) for d in (1, 2, 4, 8)])
        self.attention = nn.Linear(128, 1)
        self.context = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.decoder = Head()

    def encode(self, x):
        nb = x['neighbors'].long()
        with torch.autocast(x['xyz'].device.type, enabled=False):
            nodes, edges, mask, edge_mask = surface_features(x['xyz'], x['valid'], nb, x['edge_valid'], x['context'])
        h = self.embed(nodes) * mask[..., None]
        for block in self.blocks:
            h = block(h, edges, mask, edge_mask, nb)
        flat = h.reshape(len(h), -1, 128)
        m = mask.reshape(len(h), -1)
        a = self.attention(flat).squeeze(-1).float().masked_fill(m == 0, -1e4).softmax(-1) * m
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (flat * a[..., None]).sum(1)
        mean = (flat * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        z = torch.cat([pooled, mean, self.context(x['context'])], -1)
        return z, m.mean(1) > .05

    def forward(self, x):
        z, usable = self.encode(x)
        return self.decoder(z), z, usable


class PhysReader(nn.Module):
    """V-JEPA 2 video features + 2D tracklet graph features + timing/camera context
    -> teacher latent -> frozen teacher head."""

    def __init__(self, families=('raw_trajectory',)):
        super().__init__()
        self.families = tuple(families)
        self.rgb = load_vjepa2()
        hidden = 1024
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.context = nn.Sequential(nn.Linear(100, 32), nn.GELU())
        self.tracks = TrackEncoder(sum(CHANNELS[f] for f in self.families), len(self.families))
        self.project = nn.Sequential(nn.Linear(hidden + 256 + 32, 512), nn.GELU(), nn.Linear(512, LATENT))
        self.decoder = Head()
        self.decoder.requires_grad_(False)
        self.register_buffer('latent_mean', torch.zeros(LATENT))
        self.register_buffer('latent_std', torch.ones(LATENT))

    def init_from_teacher(self, teacher, latent_mean, latent_std):
        self.decoder.load_state_dict(teacher.decoder.state_dict())
        self.latent_mean.copy_(torch.as_tensor(latent_mean))
        self.latent_std.copy_(torch.as_tensor(latent_std).clamp_min(1e-4))

    def forward(self, x):
        video = self.norm(video_features(self.rgb, x['video']))
        tracks = self.tracks(x['tracks'])
        context = self.context(x['tracks']['metadata'].flatten(1))
        z = self.project(torch.cat([video, tracks, context], -1)).float() * self.latent_std + self.latent_mean
        return self.decoder(z), z

    def parameter_groups(self, lr=1e-4, head_lr=3e-4):
        slow = [p for m in (self.rgb, self.tracks, self.context) for p in m.parameters() if p.requires_grad]
        return [dict(params=slow, lr=lr), dict(params=list(self.project.parameters()), lr=head_lr)]


class RegressionHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(),
                                 nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)


def adapt_input(model, channels):
    """Replace the RGB patch projection with one for `channels` input channels (mean RGB filter)."""
    old = model.encoder.embeddings.patch_embeddings.proj
    new = nn.Conv3d(channels, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, channels, 1, 1, 1) * math.sqrt(3 / channels))
        new.bias.copy_(old.bias)
    model.encoder.embeddings.patch_embeddings.proj = new
    model.config.in_chans = channels


class Predictor(nn.Module):
    """Direct regression of z. rgb: V-JEPA 2 on the video; families: tracklet/cue inputs;
    extra: a second V-JEPA 2 branch on depth (1 channel) or optical flow (2 channels)."""

    def __init__(self, families=(), rgb=True, extra=None):
        super().__init__()
        self.families, self.extra_kind = tuple(families), extra
        width = 32
        if rgb:
            self.rgb = load_vjepa2()
            self.norm = nn.LayerNorm(1024, elementwise_affine=False)
            width += 1024
        self.context = nn.Sequential(nn.Linear(100, 32), nn.GELU())
        if extra:
            self.extra = load_vjepa2()
            adapt_input(self.extra, {'depth': 1, 'flow': 2}[extra])
            width += 1024
        if families:
            self.tracks = TrackEncoder(sum(CHANNELS[f] for f in families), len(families))
            width += 256
        self.heads = nn.ModuleList([RegressionHead(width) for _ in range(2)])
        self.register_buffer('center', torch.tensor([3.5, 3.1505]))
        self.register_buffer('scale', torch.tensor([2.5, 1.1505]))

    def forward(self, x):
        z = []
        if hasattr(self, 'rgb'):
            z.append(self.norm(video_features(self.rgb, x['video'])))
        z.append(self.context(x['tracks']['metadata'].flatten(1)))
        if hasattr(self, 'extra'):
            z.append(self.norm(video_features(self.extra, x['extra'])))
        if hasattr(self, 'tracks'):
            z.append(self.tracks(x['tracks']))
        z = torch.cat(z, -1)
        return torch.cat([h(z) for h in self.heads], 1).float() * self.scale + self.center, None

    def parameter_groups(self, lr=1e-4, head_lr=3e-4):
        heads = {id(p) for p in self.heads.parameters()}
        body = [p for p in self.parameters() if p.requires_grad and id(p) not in heads]
        return [dict(params=body, lr=lr), dict(params=list(self.heads.parameters()), lr=head_lr)]


MODELS = {
    'rgb': dict(cls=Predictor),
    'rgb_depth': dict(cls=Predictor, extra='depth'),
    'rgb_flow': dict(cls=Predictor, extra='flow'),
    'tracks_2d': dict(cls=Predictor, rgb=False, families=('raw_trajectory',)),
    'tracks_3d': dict(cls=Predictor, rgb=False, families=('trajectory_3d',)),
    'rgb_tracks_2d': dict(cls=Predictor, families=('raw_trajectory',)),
    'rgb_tracks_2d_cues': dict(cls=Predictor, families=('raw_trajectory', 'strain_2d', 'stretch_2d', 'nonaffine_2d')),
    'rgb_tracks_3d': dict(cls=Predictor, families=('trajectory_3d',)),
    'rgb_tracks_3d_depth_aug': dict(cls=Predictor, families=('trajectory_3d',)),
    'student_3d': dict(cls=PhysReader, families=('trajectory_3d',)),
    'physreader': dict(cls=PhysReader),
    'teacher': dict(cls=Teacher),
}


def build(name):
    spec = dict(MODELS[name])
    return spec.pop('cls')(**spec)
