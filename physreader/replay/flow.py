"""SEA-RAFT (M, Spring checkpoint) short-range optical flow between consecutive frames."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT = 'models/Tartan-C-T-TSKH-spring540x960-M.pth'


class SeaRaft:
    """`repo` is a checkout of https://github.com/princeton-vl/SEA-RAFT holding the checkpoint."""

    def __init__(self, repo, device='cuda'):
        repo = Path(repo)
        sys.path[:0] = [str(repo), str(repo / 'core')]
        from raft import RAFT
        from utils.utils import load_ckpt
        model = RAFT(Namespace(**json.loads((repo / 'config/eval/spring-M.json').read_text())))
        load_ckpt(model, str(repo / CHECKPOINT))
        self.model, self.device = model.to(device).eval(), device

    def __call__(self, a, b, iters):
        return self.model(a, b, iters=iters, test_mode=True)['flow'][-1]

    def image(self, rgb):
        return torch.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1)[None].float()

    @torch.inference_mode()
    def native(self, frames, iters, down):
        """Flow inferred at 1/down resolution, in native pixels: half-resolution fields, and native-resolution
        fields when down == 1."""
        h, w = frames[0].shape[:2]
        out = {'forward': [], 'backward': [], 'forward_native': [], 'backward_native': []}
        for t in range(len(frames) - 1):
            a, b = (F.interpolate(self.image(frames[i]), size=(h // down, w // down), mode='bilinear',
                                  align_corners=False) for i in (t, t + 1))
            for u, v, name in [(a, b, 'forward'), (b, a, 'backward')]:
                f = self(u, v, iters)
                if down == 1:
                    out[name + '_native'].append(f[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))
                half = f if down == 2 else F.interpolate(f, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
                out[name].append((down * half[0].permute(1, 2, 0).cpu().numpy()).astype(np.float16))
        return {k: np.stack(v) for k, v in out.items() if v}

    @torch.inference_mode()
    def registered(self, frames, masks, iters, zoom=2):
        """Native-resolution flow after removing the mask-centroid translation, computed in a 2x zoomed
        window around the object; the translation is added back."""
        h, w = frames[0].shape[:2]
        centers = [np.argwhere(m)[:, ::-1].mean(0) for m in masks]
        out = {'forward': [], 'backward': []}
        for t in range(len(frames) - 1):
            for src, dst, name in [(t, t + 1, 'forward'), (t + 1, t, 'backward')]:
                delta = centers[dst] - centers[src]
                shift = np.array([[1, 0, -delta[0]], [0, 1, -delta[1]]], dtype='float32')
                target = cv2.warpAffine(frames[dst], shift, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                warped = cv2.warpAffine(masks[dst].astype('uint8'), shift, (w, h), flags=cv2.INTER_NEAREST)
                yy, xx = np.where(masks[src] | warped.astype(bool))
                cx, cy = (xx.min() + xx.max()) // 2, (yy.min() + yy.max()) // 2
                rw = int(np.ceil(max(160, xx.max() - xx.min() + 96) / 8) * 8)
                rh = int(np.ceil(max(160, yy.max() - yy.min() + 96) / 8) * 8)
                x0, y0 = max(0, min(w - rw, cx - rw // 2)), max(0, min(h - rh, cy - rh // 2))
                x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
                a, b = (F.interpolate(self.image(im[y0:y1, x0:x1]), size=((y1 - y0) * zoom, (x1 - x0) * zoom),
                                      mode='bilinear', align_corners=False) for im in (frames[src], target))
                residual = self(a, b, iters)
                shift = torch.as_tensor(delta, device=self.device, dtype=torch.float32)[None, :, None, None]
                restored = F.interpolate(residual, size=(y1 - y0, x1 - x0), mode='bilinear', align_corners=False)
                restored = restored / zoom + shift
                f = torch.zeros((1, 2, h, w), device=self.device)
                f[:, :, y0:y1, x0:x1] = restored
                out[name].append(f[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))
        return {k: np.stack(v) for k, v in out.items()}

    def fields(self, frames, masks, options):
        """All flow fields the 2D observation of one video uses."""
        flow = self.native(frames, options['flow_iterations'], options['flow_downsample'])
        if options['registered_flow']:
            r = self.registered(frames, masks, options['flow_iterations'])
            flow.update(secondary_forward_native=r['forward'], secondary_backward_native=r['backward'])
        flow['indices'] = np.arange(len(frames) - 1)
        return flow
