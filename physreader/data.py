"""Reading clips. A clip is one .npz (inside a tar shard or a plain file) with:

  rgb             (16, 224, 224, 3) uint8   cropped video window C-3..C+12
  raw_trajectory  (4, 5, 128, 4)            2D tracklets: 4 windows x 5 frames x 128 points
  raw_trajectory_valid, neighbors (4, 128, 24), edge_valid, metadata (4, 5, 5)
  targets         (2,) log10 E/rho, log10 sigma_Y/rho   (synthetic only)
  yielded         plastic excitation inside the window  (synthetic only)
"""
import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)

STATS = json.loads((Path(__file__).parent / 'stats.json').read_text())
RGB_MEAN = np.array([0.485, 0.456, 0.406], 'f')[None, :, None, None]
RGB_STD = np.array([0.229, 0.224, 0.225], 'f')[None, :, None, None]
META_SCALE = np.array([1, 1 / 30, 1, 1, .01], 'f')


def video_tensor(rgb):
    x = rgb.astype('f4').transpose(0, 3, 1, 2) / 255.
    return torch.from_numpy(np.ascontiguousarray((x - RGB_MEAN) / RGB_STD))


def track_inputs(clip, families=('raw_trajectory',)):
    cues, valids = [], []
    for name in families:
        v = clip[name + '_valid'].astype(bool)
        s = STATS[name]
        x = (clip[name].astype('f4') - np.asarray(s['mean'], 'f')) / np.asarray(s['std'], 'f')
        cues.append(np.where(v[..., None], x, 0).astype('f4'))
        valids.append(v)
    family_valid = np.stack(valids, -1) if valids else np.zeros((4, 5, 128, 0), bool)
    return {
        'cue': torch.from_numpy(np.concatenate(cues, -1) if cues else np.zeros((4, 5, 128, 0), 'f4')),
        'family_valid': torch.from_numpy(family_valid),
        'valid': torch.from_numpy(family_valid.any(-1)),
        'neighbors': torch.from_numpy(clip['neighbors'].astype('int64')),
        'edge_valid': torch.from_numpy(clip['edge_valid'].astype(bool)),
        'metadata': torch.from_numpy(clip['metadata'].astype('f4') / META_SCALE),
    }


def depth_tensor(clip):
    s = STATS['depth']
    d = np.where(clip['depth_valid'] > 0, (clip['depth'].astype('f4') - s['mean']) / s['std'], 0)
    return torch.from_numpy(d[:, None].astype('f4'))


def flow_tensor(clip):
    f = np.arcsinh(clip['flow'].astype('f4') / STATS['flow']['scale']) / np.arcsinh(1.)
    return torch.from_numpy(np.ascontiguousarray(f.transpose(0, 3, 1, 2)))


def surface_inputs(clip):
    return {
        'xyz': torch.from_numpy(clip['xyz'].astype('f4')),
        'valid': torch.from_numpy(clip['valid'].astype('f4')),
        'neighbors': torch.from_numpy(clip['neighbors'].astype('int64')),
        'edge_valid': torch.from_numpy(clip['edge_valid'].astype('f4')),
        'context': torch.from_numpy(clip['context'].astype('f4')),
    }


class Clips(torch.utils.data.Dataset):
    """Clips from a folder of .npz files (e.g. PhysReader-Real inputs) or from a PhysReader-Sim index
    (split.json), whose rows point into tar shards by byte offset. Sim clips store no RGB: the
    16-frame window is cut from the source video exactly as it was for training."""

    def __init__(self, source, kind='video', families=('raw_trajectory',), extra=None, root=None):
        source = Path(source)
        if source.is_dir():
            self.rows = [{'key': p.stem, 'file': p.name} for p in sorted(source.glob('*.npz'))]
        else:
            self.rows = json.loads(source.read_text())
            if kind == 'surface':
                self.rows = [r for r in self.rows if 'surface' in r]
        self.root = Path(root) if root else (source if source.is_dir() else source.parent)
        self.kind, self.families, self.extra = kind, families, extra

    def __len__(self):
        return len(self.rows)

    def read(self, ref):
        with open(self.root / ref['file'], 'rb') as f:
            if 'offset' not in ref:
                return f.read()
            f.seek(ref['offset'])
            return f.read(ref['size'])

    def __getitem__(self, i):
        row = self.rows[i]
        ref = row.get('surface' if self.kind == 'surface' else 'clips', row)
        with np.load(io.BytesIO(self.read(ref))) as z:
            clip = {k: z[k] for k in z.files}
        if self.kind == 'surface':
            x = surface_inputs(clip)
        else:
            rgb = clip['rgb'] if 'rgb' in clip else rgb_window(self.read(row['video']), row['frames'], row['crop'])
            x = {'video': video_tensor(rgb), 'tracks': track_inputs(clip, self.families)}
            if self.extra == 'depth':
                x['extra'] = depth_tensor(clip)
            elif self.extra == 'flow':
                x['extra'] = flow_tensor(clip)
        if 'targets' in clip:
            x['target'] = torch.from_numpy(clip['targets'].astype('f4'))
            x['yielded'] = torch.tensor(float(clip['yielded']))
        elif 'target' in row:
            x['target'] = torch.tensor(row['target'], dtype=torch.float32)
            x['yielded'] = torch.tensor(float(row['yielded']))
        x['key'] = row.get('uid', row.get('key'))
        return x


def rgb_window(mp4, frames, crop, size=224):
    """Decode the source video and resample the square crop (left, top, side) to size x size."""
    left, top, side = crop[:3]
    grid = (np.arange(size, dtype='f') + .5) * side / size - .5
    mx, my = np.meshgrid(grid + left, grid + top)
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        tmp.write(mp4)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        out = []
        for i in range(max(frames) + 1):
            ok, im = cap.read()
            if not ok:
                raise IOError(f'could not read frame {i}')
            if i in frames:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                out.append(cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        cap.release()
    return np.stack(out)

CHANNELS = {'raw_trajectory': 4, 'strain_2d': 6, 'stretch_2d': 8, 'nonaffine_2d': 2, 'stretch_3d': 8, 'trajectory_3d': 6}
