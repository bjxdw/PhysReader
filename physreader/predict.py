"""Predict z = (log10 E/rho, log10 sigma_Y/rho) for prepared clips and score them on PhysReader-Real.

    python -m physreader.predict --model physreader --checkpoint physreader.pt \
        --inputs PhysReader-Real/inputs --real PhysReader-Real --out real.json
    python -m physreader.predict --model physreader --checkpoint physreader.pt \
        --inputs PhysReader-Sim/val.json --out val.json
"""
import argparse
import json
from pathlib import Path

import torch

from .data import Clips
from .metrics import real_scores, synthetic_mae
from .models import MODELS, build


def load(name, checkpoint):
    model = build(name)
    state = torch.load(checkpoint, map_location='cpu')['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not unexpected and not set(missing) & trainable, (unexpected, set(missing) & trainable)
    return model.cuda().eval()


@torch.no_grad()
def predict(model, clips, batch_size=4):
    out = {}
    for x in torch.utils.data.DataLoader(clips, batch_size=batch_size, num_workers=4):
        keys = x.pop('key')
        x = {k: ({a: b.cuda() for a, b in v.items()} if isinstance(v, dict) else v.cuda()) for k, v in x.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(x)[0]
        out.update(zip(keys, z.float().cpu().tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS), required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--inputs', required=True, help='folder of .npz clips or an index .json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--real', help='PhysReader-Real folder: also report the parameter distance')
    a = ap.parse_args()

    spec = MODELS[a.model]
    kind = 'surface' if a.model == 'teacher' else 'video'
    clips = Clips(a.inputs, kind, spec.get('families', ('raw_trajectory',)), spec.get('extra'))
    pred = predict(load(a.model, a.checkpoint), clips)
    result = {'predictions': pred}
    rows = [r for r in clips.rows if 'target' in r]
    if rows:
        result['synthetic'] = synthetic_mae([pred[r['uid']] for r in rows], [r['target'] for r in rows], [r['yielded'] for r in rows])
        print(f"synthetic MAE over {len(rows)} clips: {result['synthetic']}")
    if a.real:
        root = Path(a.real)
        drops = {d['key']: d for d in json.loads((root / 'drops.json').read_text())}
        regions = json.loads((root / 'regions.json').read_text())
        result['parameter_distance'], result['per_drop'] = real_scores(pred, drops, regions)
        print(f"parameter distance over {len(pred)} drops: {result['parameter_distance']:.3f}")
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
