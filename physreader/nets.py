"""Graph networks over tracked points: 2D tracklets (students, direct models) and
exact surface particles (teacher)."""
import torch
from torch import nn
import torch.nn.functional as F


def gather_neighbors(x, nb):
    """x (B, T, N, D), nb (B, N, K) -> (B, T, N, K, D)."""
    b, t, n, d = x.shape
    k = nb.shape[-1]
    idx = nb[:, None].expand(b, t, n, k).reshape(b, t, n * k)[..., None].expand(-1, -1, -1, d)
    return torch.gather(x, 2, idx).reshape(b, t, n, k, d)


def masked_max(h, valid, dim):
    """Max over `dim` of h (..., C) where valid (...) is true; 0 where nothing is valid."""
    m = h.masked_fill(~valid[..., None], -1e4).max(dim).values
    return torch.where(valid.any(dim if dim >= 0 else dim + 1)[..., None], m, 0.)


def temporal_conv(conv, h, valid, dilation):
    """Dilated temporal convolution along each track, rescaled by the number of valid taps."""
    b, t, n, c = h.shape
    z = h.permute(0, 2, 3, 1).reshape(b * n, c, t)
    m = valid.permute(0, 2, 1).reshape(b * n, 1, t)
    m = m.to(z.dtype) if m.dtype == torch.bool else m
    ones = torch.ones(1, 1, 3, device=z.device, dtype=m.dtype)
    count = F.conv1d(m, ones, padding=dilation, dilation=dilation).clamp_min(1)
    z = conv(z) * (3 / count)
    return z.reshape(b, n, c, t).permute(0, 3, 1, 2)


class TrackBlock(nn.Module):
    """EdgeConv over the fixed k-NN graph, then a dilated temporal convolution."""

    def __init__(self, dilation, width=128):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dilation = dilation

    def forward(self, h, valid, nb, edge_valid):
        hj = gather_neighbors(h, nb)
        hi = h[:, :, :, None].expand_as(hj)
        edges = valid[:, :, :, None] & gather_neighbors(valid[..., None], nb)[..., 0] & edge_valid[:, None]
        msg = self.message(torch.cat([hi, hj - hi], -1))
        h = (h + F.gelu(masked_max(msg, edges, -2))) * valid[..., None]
        z = temporal_conv(self.temporal, h, valid, self.dilation)
        return (h + F.gelu(self.norm(z))) * valid[..., None]


class TrackEncoder(nn.Module):
    """Encodes per-point features on 4 windows x 5 frames x 128 points into a 256-d vector."""

    def __init__(self, channels, families=1, width=128):
        super().__init__()
        self.channels = channels
        self.projection = nn.Sequential(nn.Linear(channels + families + 5, width), nn.GELU())
        self.blocks = nn.ModuleList([TrackBlock(d, width) for d in (1, 2, 4)])
        self.width = width

    def forward(self, x):
        cue = x['cue']
        b, w, t, n, c = cue.shape
        valid = x['valid'].reshape(b * w, t, n)
        nb = x['neighbors'].reshape(b * w, n, -1)
        edge_valid = x['edge_valid'].reshape(b * w, n, -1)
        meta = x['metadata'][:, :, :, None].expand(b, w, t, n, 5)
        feats = torch.cat([cue, x['family_valid'].to(cue.dtype), meta], -1)
        h = self.projection(feats.reshape(b * w, t, n, -1)) * valid[..., None]
        for block in self.blocks:
            h = block(h, valid, nb, edge_valid)
        h = h.reshape(b, w * t * n, self.width)
        valid = valid.reshape(b, w * t * n)
        mean = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return torch.cat([mean, masked_max(h, valid, 1)], -1)


@torch.no_grad()
def surface_features(x, valid, nb, edge_valid, context):
    """Node and edge features of exact surface particles: local tangent-plane strain,
    strain rate, non-affine residual, bending and edge stretch."""
    x, valid = x.float(), valid.float()
    b, t, n, _ = x.shape
    edges = gather_neighbors(x, nb) - x[:, :, :, None]
    w = valid[:, :, :, None] * gather_neighbors(valid[..., None], nb)[..., 0] * edge_valid[:, None].float()
    ref, w0 = edges[:, 0], w[:, 0]

    ev, U = torch.linalg.eigh(torch.einsum('bnki,bnk,bnkj->bnij', ref, w0, ref))
    basis, normal0 = U[..., 1:], U[..., 0]
    planar = (ev[..., 1] > 1e-8) & (ev[..., 1] > ev[..., 2] * .01) & (w0.sum(-1) >= 6)
    A = torch.einsum('bnki,bnij->bnkj', ref, basis)
    eig, V = torch.linalg.eigh(torch.einsum('bnki,btnk,bnkj->btnij', A, w, A))
    good = planar[:, None] & (w.sum(-1) >= 6) & (eig[..., 0] > 1e-8) & (eig[..., 0] > eig[..., 1] * .01)
    inv = (V * torch.where(good[..., None], eig.clamp_min(1e-8).reciprocal(), 0.)[..., None, :]) @ V.transpose(-1, -2)
    Fs = torch.einsum('btnki,btnk,bnkj->btnij', edges, w, A) @ inv

    strain = .5 * torch.linalg.eigvalsh(Fs.transpose(-1, -2) @ Fs).clamp_min(1e-10).log()
    good = good & torch.isfinite(strain).all(-1)
    strain = torch.where(good[..., None], strain, 0.)
    area = strain.sum(-1, keepdim=True)
    shear = (strain[..., 1:] - strain[..., :1]).abs()
    dt = context[:, 1].exp().view(b, 1, 1, 1)
    rates = torch.cat([torch.zeros_like(strain[:, :1]), (strain[:, 1:] - strain[:, :-1]) / dt], 1)
    rates *= torch.cat([torch.zeros_like(good[:, :1]), good[:, 1:] & good[:, :-1]], 1)[..., None]

    fit = torch.einsum('btnij,bnkj->btnki', Fs, A)
    scale = (torch.sum(ref.square().sum(-1) * w0, -1) / w0.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-5)
    residual = (torch.sum((edges - fit).square().sum(-1) * w, -1) / w.sum(-1).clamp_min(1)).sqrt() / scale[:, None]

    _, Ut = torch.linalg.eigh(torch.einsum('btnki,btnk,btnkj->btnij', edges, w, edges))
    normals = Ut[..., 0]
    cos0 = (normal0[:, :, None] * gather_neighbors(normal0[:, None], nb)[:, 0]).sum(-1).abs().clamp(0, 1)
    cost = (normals[:, :, :, None] * gather_neighbors(normals, nb)).sum(-1).abs().clamp(0, 1)
    bend = torch.acos(cost) - torch.acos(cos0[:, None])
    edge_good = w * good[..., None] * gather_neighbors(good[..., None].float(), nb)[..., 0]
    bend_mean = (bend.abs() * edge_good).sum(-1) / edge_good.sum(-1).clamp_min(1)

    l0 = ref.norm(dim=-1).clamp_min(1e-6)
    stretch = (edges.norm(dim=-1).clamp_min(1e-6) / l0[:, None]).log()
    nodes = torch.cat([
        torch.asinh(strain / .05), torch.asinh(area / .1), torch.asinh(shear / .05), torch.asinh(rates / 1.5),
        torch.asinh(rates.sum(-1, keepdim=True) / 3), torch.asinh(bend_mean[..., None] / .1),
        torch.asinh(residual[..., None] / .05), valid[..., None]], -1)
    edge_feats = torch.stack([
        l0[:, None].expand(-1, t, -1, -1).log(), torch.asinh(stretch / .05),
        cos0[:, None].expand(-1, t, -1, -1), torch.asinh(bend / .1), edge_good], -1)
    nodes = torch.nan_to_num(nodes) * good[..., None]
    edge_feats = torch.nan_to_num(edge_feats) * edge_good[..., None]
    return nodes, edge_feats, good.float(), edge_good


class SurfaceBlock(nn.Module):
    def __init__(self, dilation, width=128):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * width + 5, width), nn.LayerNorm(width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.project = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU())
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dilation = dilation

    def forward(self, h, edge, mask, edge_mask, nb):
        hj = gather_neighbors(h, nb)
        hi = h[:, :, :, None].expand_as(hj)
        msg = self.message(torch.cat([hi, hj - hi, edge], -1))
        mean = (msg * edge_mask[..., None]).sum(-2) / edge_mask.sum(-1, keepdim=True).clamp_min(1)
        h = (h + self.project(torch.cat([mean, masked_max(msg, edge_mask > 0, -2)], -1))) * mask[..., None]
        z = temporal_conv(self.temporal, h, mask, self.dilation)
        return (h + F.gelu(self.norm(z))) * mask[..., None]
