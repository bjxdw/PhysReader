"""Training (single or multi GPU via torchrun). Global batch 480 at every GPU count.

    torchrun --nproc_per_node 8 -m physreader.train --model teacher --data data/PhysReader-Sim --out runs/teacher
    torchrun --nproc_per_node 8 -m physreader.train --model physreader --teacher runs/teacher/best.pt ...
"""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import Clips
from .metrics import synthetic_mae
from .models import MODELS, build

GLOBAL_BATCH, GROUP, SCALE = 480, 24, torch.tensor([2.5, 1.1505])
# these need per-clip dense flow or depth, which PhysReader-Sim does not include
NOT_RELEASED = {'rgb_flow', 'rgb_depth', 'rgb_tracks_3d_depth_aug'}

# epochs, epochs of cosine decay (then held), final LR fraction, weight decay, grad clip
SCHEDULES = {'teacher': (10, 5, .1, 1e-4, 5.), 'default': (5, 4, .05, .05, 1.)}


class GlobalBatches(torch.utils.data.Sampler):
    """Each update uses 480 clips split over all ranks. The loss is the mean over groups of 24 clips of
    the per-group mean, with yield stress averaged only over clips whose window yields. Ranks with
    nothing left receive zero-weight padding, so the update is the same for any number of GPUs."""

    def __init__(self, yielded, rank, world, epoch, batch=4):
        self.yielded, self.rank, self.world, self.epoch, self.batch = yielded, rank, world, epoch, batch
        self.micro = math.ceil(GLOBAL_BATCH / (world * batch))

    def __len__(self):
        return math.ceil(len(self.yielded) / GLOBAL_BATCH) * self.micro

    def __iter__(self):
        order = np.random.default_rng(1701 + self.epoch).permutation(len(self.yielded)).tolist()
        slots = self.micro * self.world * self.batch
        for start in range(0, len(order), GLOBAL_BATCH):
            ids = order[start:start + GLOBAL_BATCH]
            n, groups = len(ids), math.ceil(len(ids) / GROUP)
            den = []
            for g in range(0, GLOBAL_BATCH, GROUP):
                members = ids[g:g + GROUP]
                ny = sum(self.yielded[i] for i in members)
                den += [(max(1, len(members)) * groups, max(1, ny * groups))] * GROUP
            den += [(GLOBAL_BATCH, GLOBAL_BATCH)] * (slots - GLOBAL_BATCH)
            ids = ids + [ids[0]] * (slots - n)
            for m in range(self.micro):
                first = m * self.world * self.batch + self.rank * self.batch
                yield [(ids[j], float(j < n), *den[j]) for j in range(first, first + self.batch)]


class Weighted(torch.utils.data.Dataset):
    def __init__(self, clips, latents=None):
        self.clips, self.latents = clips, latents

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        i, weight, den_e, den_y = item
        x = self.clips[i]
        x['weight'], x['den_E'], x['den_Y'] = torch.tensor(weight), torch.tensor(den_e), torch.tensor(den_y)
        if self.latents is not None:
            j = self.latents['index'].get(x['key'])
            x['latent'] = torch.from_numpy(self.latents['latent'][j]) if j is not None else torch.zeros(288)
            x['latent_valid'] = torch.tensor(j is not None and bool(self.latents['valid'][j]))
        return x


def loss_fn(kind, pred, z, x, model, world):
    w = x['weight']
    if kind == 'teacher':
        err = F.smooth_l1_loss(pred, x['target'], beta=.25, reduction='none')
        w = w * x['usable']
    else:
        err = ((pred - x['target']) / SCALE.to(pred.device)).square()
    loss = (err[:, 0] * w / x['den_E']).sum() + (err[:, 1] * w * x['yielded'] / x['den_Y']).sum()
    if kind == 'physreader':
        lat = F.smooth_l1_loss((z - x['latent']) / model.latent_std, torch.zeros_like(z), beta=.25, reduction='none').mean(-1)
        loss = loss + (lat * x['latent_valid'] * x['weight'] / x['den_E']).sum()
    return world * loss


def to_device(x):
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    return x.cuda(non_blocking=True) if torch.is_tensor(x) else x


@torch.no_grad()
def evaluate(model, clips, rank, world):
    model.eval()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(clips, range(rank, len(clips), world)), batch_size=4, num_workers=4)
    rows = []
    for x in loader:
        keys = x.pop('key')
        x = to_device(x)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(x)[0]
        rows += [dict(key=k, pred=p, target=t, yielded=y) for k, p, t, y in
                 zip(keys, pred.float().tolist(), x['target'].tolist(), x['yielded'].tolist())]
    parts = [None] * world
    dist.all_gather_object(parts, rows)
    rows = [r for p in parts for r in p]
    return rows, synthetic_mae([r['pred'] for r in rows], [r['target'] for r in rows], [r['yielded'] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=[m for m in MODELS if m not in NOT_RELEASED], required=True)
    ap.add_argument('--data', required=True, help='folder with train.json / val.json clip indices')
    ap.add_argument('--out', required=True)
    ap.add_argument('--teacher', help='teacher checkpoint (PhysReader / student_3d)')
    ap.add_argument('--latents', help='teacher latents .npz from physreader.export_latents')
    ap.add_argument('--seed', type=int, default=1701)
    a = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    kind = 'teacher' if a.model == 'teacher' else 'physreader' if MODELS[a.model]['cls'].__name__ == 'PhysReader' else 'direct'
    spec = MODELS[a.model]
    families = spec.get('families', ('raw_trajectory',))
    data_kind = 'surface' if kind == 'teacher' else 'video'
    train = Clips(Path(a.data) / 'train.json', data_kind, families, spec.get('extra'))
    val = Clips(Path(a.data) / 'val.json', data_kind, families, spec.get('extra'))

    model = build(a.model).cuda()
    latents = None
    if kind == 'physreader':
        teacher = torch.load(a.teacher, map_location='cpu')
        model.decoder.load_state_dict({k[len('decoder.'):]: v for k, v in teacher['model'].items() if k.startswith('decoder.')})
        z = np.load(a.latents)
        model.latent_mean.copy_(torch.from_numpy(z['mean']))
        model.latent_std.copy_(torch.from_numpy(z['std']).clamp_min(1e-4))
        latents = {'index': {str(k): i for i, k in enumerate(z['keys'])}, 'latent': z['latent'].astype('f4'), 'valid': z['valid']}

    epochs, decay_epochs, floor, wd, clip = SCHEDULES['teacher' if kind == 'teacher' else 'default']
    groups = [dict(params=list(model.parameters()), lr=3e-4)] if kind == 'teacher' else model.parameter_groups()
    opt = torch.optim.AdamW(groups, weight_decay=wd, eps=1e-8)
    updates = math.ceil(len(train) / GLOBAL_BATCH)
    total, warm = updates * decay_epochs, max(1, int(.05 * updates * decay_epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))))
    ddp = DDP(model, device_ids=[torch.cuda.current_device()])
    yielded = [bool(r['yielded']) for r in train.rows]
    data = Weighted(train, latents)
    best = float('inf')

    for epoch in range(epochs):
        model.train()
        sampler = GlobalBatches(yielded, rank, world, epoch)
        loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
        opt.zero_grad()
        for i, x in enumerate(loader):
            x = to_device(x)
            step = (i + 1) % sampler.micro == 0
            with contextlib.nullcontext() if step else ddp.no_sync():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out_ = ddp(x)
                    if kind == 'teacher':
                        pred, z, x['usable'] = out_
                    else:
                        pred, z = out_
                    loss = loss_fn(kind, pred, z, x, model, world)
                loss.backward()
            if step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                if rank == 0 and (i + 1) // sampler.micro % 50 == 0:
                    print(f'epoch {epoch + 1} update {(i + 1) // sampler.micro}/{updates} loss {loss.item():.4f}', flush=True)
        rows, metrics = evaluate(model, val, rank, world)
        if rank == 0:
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            state = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable or k.startswith('decoder.') or k.startswith('latent_')}
            ckpt = {'model': state, 'epoch': epoch + 1, 'model_name': a.model, 'val': metrics}
            torch.save(ckpt, out / f'epoch{epoch + 1:03d}.pt')
            score = metrics['E_mae'] + (metrics['YS_mae'] or 0)
            if score < best:
                best = score
                torch.save(ckpt, out / 'best.pt')
            (out / f'val_epoch{epoch + 1:03d}.json').write_text(json.dumps({'metrics': metrics, 'predictions': rows}))
            print(f'epoch {epoch + 1} val {metrics}', flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
