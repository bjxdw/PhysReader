"""Replay predicted materials on PhysReader-Real (Table 1 replay columns).

    python -m physreader.replay --data PhysReader-Real --predictions real.json --sea-raft SEA-RAFT --out replay
    python -m physreader.replay --data PhysReader-Real --reference --fixed --sea-raft SEA-RAFT --out replay_reference

For each drop the material is held fixed and 25 initial conditions are tried over the window C-3..C+12
(--fixed: only the reference's selected one). The trial with the lowest loss is compared with the
reference's replay loss (delta total and delta deformation, in percent). Err_ref simulates the material
from the reference's initial state and compares it particle by particle with the reference simulation.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import motion, observe, points
from .data import Drop, decode, drops
from .flow import SeaRaft
from .search import deformation, prior, search, total
from .simulate import simulate


def plain(x):
    return x.tolist() if isinstance(x, np.ndarray) else x.item()


def trial(drop, raft, E, yield_stress, eta, video):
    """Loss terms of one simulated drop (E, yield stress in Pa; initial state eta)."""
    sim = simulate(drop, E, yield_stress, eta, video=video)
    flow = raft.fields(sim['frames'], sim['masks'], drop.options)
    length = drop.config['length_px']
    signals = observe.describe(flow, sim['masks'], length, drop.options)
    m = observe.score(signals, drop.config['real'], length)
    m.update(motion.score(drop.config['real_motion'], motion.surface_motion(drop, sim), sim['masks'], drop.real_masks))
    return m


def replay(drop, raft, E, yield_stress, trials=25, fixed=False):
    with tempfile.TemporaryDirectory() as tmp:
        evaluate = lambda i, eta: trial(drop, raft, E, yield_stress, eta, Path(tmp) / f'{i:02d}.mp4')
        if fixed:
            eta = drop.reference['initial_state']
            m = evaluate(0, eta)
            m['prior'] = prior(eta)
            result = dict(selected=dict(trial=0, initial_state=eta, metrics=m), trials=[], failures=[])
        else:
            c = drop.config
            result = search(evaluate, drop.name, c['state'], c['anchor'], c['tolerance'], trials)
    m, ref = result['selected']['metrics'], drop.config['reference_metrics']
    result.update(delta_total=100 * (total(m) / total(ref) - 1),
                  delta_deformation=100 * (deformation(m) / deformation(ref) - 1))
    return result


def err_ref(drop, E, yield_stress):
    return points.err_ref(simulate(drop, E, yield_stress, drop.config['state'])['positions'], drop.reference_tracks())


def materials(args, keys):
    """E and yield stress in Pa for each drop."""
    rows = {d['key']: d for d in json.loads((Path(args.data) / 'drops.json').read_text())}
    if args.reference:
        return {k: (rows[k]['reference']['E_Pa'], rows[k]['reference']['YS_Pa']) for k in keys}
    pred = json.loads(Path(args.predictions).read_text())
    pred = {k.replace('/', '_'): z for k, z in pred.get('predictions', pred).items()}
    return {k: decode(pred[k], rows[k]['density']) for k in keys}


def summarize(rows):
    out = {'drops': len(rows)}
    for name in ['delta_total', 'delta_deformation']:
        values = [r['replay'][name] for r in rows if 'replay' in r]
        out[name] = float(np.mean(values)) if values else None
    err = [r for r in rows if 'err_ref' in r]
    if err:
        objects = {}
        for r in err:
            objects.setdefault(r['key'].rsplit('_', 1)[0], []).append(r['err_ref']['total_with_point_fit_strain'])
        out['err_ref'] = float(np.mean([np.mean(v) for v in objects.values()]))
        out['err_ref_drop_mean'] = float(np.mean([v for o in objects.values() for v in o]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='PhysReader-Real folder')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', help='{drop: [log10 E/rho, log10 sigma_Y/rho]}, e.g. physreader.predict output')
    group.add_argument('--reference', action='store_true', help='replay the reference materials')
    ap.add_argument('--sea-raft', help='SEA-RAFT checkout holding the Spring-M checkpoint (needed for the replay)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--drops', nargs='*', help='subset of drop keys, e.g. box_1')
    ap.add_argument('--trials', type=int, default=25)
    ap.add_argument('--fixed', action='store_true', help="only the reference's selected initial state")
    ap.add_argument('--skip-replay', action='store_true')
    ap.add_argument('--skip-err-ref', action='store_true')
    a = ap.parse_args()

    keys = a.drops or drops(a.data)
    mats = materials(a, keys)
    raft = None if a.skip_replay else SeaRaft(a.sea_raft)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        path = out / f'{key}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        drop, (E, Y), t = Drop(a.data, key), mats[key], time.time()
        row = dict(key=key, E_Pa=E, YS_Pa=Y)
        if not a.skip_replay:
            row['replay'] = replay(drop, raft, E, Y, a.trials, a.fixed)
            r = row['replay']
            print(f"{key}: delta total {r['delta_total']:+.1f}%, delta deformation {r['delta_deformation']:+.1f}%")
        if not a.skip_err_ref:
            row['err_ref'] = err_ref(drop, E, Y)
            print(f"{key}: Err_ref {row['err_ref']['total_with_point_fit_strain']:.4f}")
        row['seconds'] = time.time() - t
        path.write_text(json.dumps(row, indent=1, default=plain))
        rows.append(row)
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
