"""Evaluation in z = (log10 E/rho, log10 sigma_Y/rho), in dex."""
import numpy as np


def synthetic_mae(pred, target, yielded):
    """Stiffness MAE over all clips; yield-stress MAE over clips whose window yields."""
    err = np.abs(np.asarray(pred) - np.asarray(target))
    y = np.asarray(yielded) > .5
    return {'E_mae': float(err[:, 0].mean()), 'YS_mae': float(err[y, 1].mean()) if y.any() else None}


def _in_triangle(p, t):
    u, v = np.linalg.solve(np.stack([t[1] - t[0], t[2] - t[0]], 1), p - t[0])
    return u >= -1e-10 and v >= -1e-10 and u + v <= 1 + 1e-10


def _nearest_on_segment(p, a, b):
    d = b - a
    dd = d @ d
    return a + np.clip((p - a) @ d / dd, 0, 1) * d if dd > 1e-24 else a.copy()


def load_regions(regions):
    """regions.json -> numpy arrays."""
    out = {}
    for name, g in regions.items():
        g = dict(g)
        for k in ('accepted', 'segments', 'triangles'):
            g[k] = np.asarray(g[k], dtype=float)
        out[name] = g
    return out


def parameter_distance(z, region):
    """Distance from a prediction to an object's plausible region (0 inside).
    Objects that never yield are scored on stiffness only."""
    z = np.asarray(z, dtype=float)
    if not region['yielded']:
        return float(abs(z[0] - np.clip(z[0], *region['E_interval'])))
    for t in region['triangles']:
        if _in_triangle(z, t):
            return 0.
    candidates = list(region['accepted'])
    candidates += [_nearest_on_segment(z, a, b) for a, b in region['segments']]
    candidates += [_nearest_on_segment(z, t[j], t[(j + 1) % 3]) for t in region['triangles'] for j in range(3)]
    return float(np.linalg.norm(np.asarray(candidates) - z, axis=1).min())


def real_scores(predictions, drops, regions):
    """predictions: {drop: z}; drops: {drop: {'object': ...}}; mean over drops."""
    regions = load_regions(regions)
    per_drop = {k: parameter_distance(z, regions[drops[k]['object']]) for k, z in predictions.items()}
    return float(np.mean(list(per_drop.values()))), per_drop
