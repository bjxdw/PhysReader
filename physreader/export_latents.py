"""Teacher latents for training PhysReader: one 288-d latent per clip, computed from the clip's exact
surface particles. Clips without teacher input get a zero latent marked invalid. mean/std are over the
valid training latents.

    python -m physreader.export_latents --teacher runs/teacher/best.pt --data data/PhysReader-Sim --out latents
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data import Clips
from .models import Teacher


@torch.no_grad()
def encode(teacher, clips):
    refs = {}
    for i, row in enumerate(clips.rows):
        refs.setdefault((row['surface']['file'], row['surface']['offset']), i)
    unique = torch.utils.data.Subset(clips, list(refs.values()))
    out = {}
    for x in torch.utils.data.DataLoader(unique, batch_size=4, num_workers=4):
        keys = x.pop('key')
        x = {k: v.cuda() for k, v in x.items() if torch.is_tensor(v)}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, valid = teacher.encode(x)
        for k, a, b in zip(keys, z.float().cpu().numpy(), valid.cpu().numpy()):
            out[k] = a, bool(b)
    by_ref = {ref: out[clips.rows[i]['uid']] for ref, i in refs.items()}
    return {r['uid']: by_ref[(r['surface']['file'], r['surface']['offset'])] for r in clips.rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--data', required=True, help='PhysReader-Sim folder')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    teacher = Teacher()
    teacher.load_state_dict(torch.load(a.teacher, map_location='cpu')['model'])
    teacher = teacher.cuda().eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = None
    for split in ('train', 'val'):
        clips = Clips(Path(a.data) / f'{split}.json', 'surface')
        found = encode(teacher, clips)
        keys = [r['uid'] for r in Clips(Path(a.data) / f'{split}.json').rows]
        latent = np.stack([found[k][0] if k in found else np.zeros(288, 'f4') for k in keys]).astype('f4')
        valid = np.array([k in found and found[k][1] for k in keys])
        if stats is None:
            z = latent[valid].astype('f8')
            stats = z.mean(0).astype('f4'), np.maximum(z.std(0), 1e-4).astype('f4')
        np.savez_compressed(out / f'{split}.npz', keys=np.asarray(keys), latent=latent, valid=valid, mean=stats[0], std=stats[1])
        print(split, len(keys), 'clips,', int(valid.sum()), 'valid latents', flush=True)


if __name__ == '__main__':
    main()
