"""2D observation of a drop video and its comparison with the real drop: mask trajectory and principal axis,
boundary non-affine flow ("affine thickness") and short flow tracklets ("short strain")."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from .data import C, FRAMES

FB_LIMIT = 2.
cv2.setNumThreads(1)


def sample(a, q):
    if len(q) == 0:
        return np.empty((0,) + np.asarray(a).shape[2:], dtype='float32')
    return cv2.remap(a.astype('float32'), q[:, 0].astype('float32')[None], q[:, 1].astype('float32')[None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]


def window_starts(stride):
    return [t for t in range(C - 4, C + 10, stride) if 0 <= t and t + 4 < FRAMES]


def mask_geometry(masks):
    out = {k: [] for k in ('center', 'angle', 'anisotropy', 'extent')}
    for m in masks:
        y, x = np.where(m)
        if len(x) < 20:
            raise ValueError('empty or tiny foreground mask')
        q = np.c_[x, y]
        v, Q = np.linalg.eigh(np.cov(q.T))
        out['center'].append(q.mean(0))
        out['angle'].append(np.arctan2(Q[1, -1], Q[0, -1]))
        out['anisotropy'].append((v[1] - v[0]) / max(v.sum(), 1))
        out['extent'].append([np.ptp(x), np.ptp(y), len(x)])
    return {k: np.array(v) for k, v in out.items()}


def robust_fit(A, y):
    beta = np.linalg.lstsq(A, y, rcond=1e-6)[0]
    for _ in range(4):
        e = np.linalg.norm(y - A @ beta, axis=-1)
        s = max(.15, float(np.median(e)) * 1.4826)
        w = np.sqrt(np.minimum(1, 1.5 * s / np.maximum(e, 1e-6)))
        beta = np.linalg.lstsq(A * w[:, None], y * w[:, None], rcond=1e-6)[0]
    return beta


def nonaffine(flow, mask, length):
    """Mask crop, distance to the boundary, |flow - robust affine fit| / length and the perimeter."""
    yy, xx = np.where(mask)
    if len(xx) < 24:
        raise ValueError('mask has fewer than 24 half-resolution pixels')
    x0, x1 = max(0, xx.min() - 5), min(mask.shape[1], xx.max() + 6)
    y0, y1 = max(0, yy.min() - 5), min(mask.shape[0], yy.max() + 6)
    m, f = mask[y0:y1, x0:x1], flow[y0:y1, x0:x1].astype(float)
    den = gaussian_filter(m.astype(float), .75)
    fs = np.stack([gaussian_filter(f[..., i] * m, .75) / np.maximum(den, 1e-6) for i in range(2)], -1)
    dist = cv2.distanceTransform(m.astype('uint8'), cv2.DIST_L2, 5) * 2
    fit = dist > 4
    if fit.sum() < 24:
        fit = m
    y, x = np.where(fit)
    ix = np.linspace(0, len(x) - 1, min(2000, len(x))).astype(int)
    x, y = x[ix], y[ix]
    cy, cx = np.indices(m.shape)
    A = np.stack([np.ones(m.shape), 2 * (cx - x.mean()) / length, 2 * (cy - y.mean()) / length], -1)
    mag = np.linalg.norm(fs - A @ robust_fit(A[y, x], fs[y, x]), axis=-1) / length
    contours, _ = cv2.findContours(m.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return m, dist, mag, max(2 * sum(cv2.arcLength(c, True) for c in contours), 2.)


def advect(q, start, length, fw, bw, masks, divisor):
    """Forward-backward checked tracklets from points q in frame `start`."""
    uv, valid = [q.copy()], [np.ones(len(q), bool)]
    h, w = masks.shape[1:]
    for t in range(start, start + length - 1):
        f = sample(fw[t], uv[-1] / divisor)
        n = uv[-1] + f
        e = np.linalg.norm(f + sample(bw[t], n / divisor), axis=1)
        interior = cv2.erode(masks[t + 1].astype('uint8'), np.ones((3, 3), np.uint8))
        valid.append(valid[-1] & (n[:, 0] > 1) & (n[:, 0] < w - 2) & (n[:, 1] > 1) & (n[:, 1] < h - 2)
                     & (sample(interior, n) > .99) & (e < FB_LIMIT))
        uv.append(n)
    return np.array(uv), np.array(valid)


def pair(flow, name):
    return flow[name.format('forward')].astype('float32'), flow[name.format('backward')].astype('float32')


def describe(flow, masks, length, options):
    """2D signals of one video from its flow fields and object masks."""
    fw, bw = pair(flow, '{}')
    curves = []
    for k, t in enumerate(flow['indices']):
        m = cv2.resize(masks[t].astype('uint8'), (fw.shape[2], fw.shape[1]), interpolation=cv2.INTER_NEAREST)
        m, dist, mag, perimeter = nonaffine(fw[k], m.astype(bool), length)
        band = m & (dist > 2) & (dist <= 2 + .08 * length)
        weight = np.maximum((expit((mag - .01) / .0025) - expit(-4)) / (1 - expit(-4)), 0)
        curves.append(float(weight[band].sum() * 4 / perimeter / length))
    native, second = 'forward_native' in flow, 'secondary_forward_native' in flow
    tfw, tbw = pair(flow, '{}_native') if native else (fw, bw)
    if second:
        sfw, sbw = pair(flow, 'secondary_{}_native')
    erode, spacing = options['track_erode'], options['track_spacing']
    windows = []
    for start in window_starts(options['window_stride']):
        yy, xx = np.where(cv2.erode(masks[start].astype('uint8'), np.ones((erode, erode), np.uint8)))
        pick = (xx % spacing == 0) & (yy % spacing == 0)
        q = np.c_[xx[pick], yy[pick]].astype(float)
        if len(q) > 220:
            q = q[np.linspace(0, len(q) - 1, 220).astype(int)]
        uv, valid = advect(q, start, 5, tfw, tbw, masks, 1 if native else 2)
        if second:
            uv2, valid2 = advect(q, start, 5, sfw, sbw, masks, 1)
            if int(valid2[2:4].sum()) > int(valid[2:4].sum()):
                uv, valid = uv2, valid2
        rates, coverage = [], []
        for lag in [2, 3, 4]:
            ids = np.flatnonzero(valid[lag])
            area = float(np.prod(np.ptp(q[ids], axis=0)) / max(np.prod(np.ptp(q, axis=0)), 1)) if len(ids) > 2 else 0.
            coverage.append(area)
            if len(ids) < 8:
                rates.append(None)
                continue
            i, j = np.triu_indices(len(ids), 1)
            d0 = np.linalg.norm(q[ids][i] - q[ids][j], axis=1)
            use = (d0 >= 12) & (d0 <= .35 * length)
            i, j, d0 = i[use], j[use], d0[use]
            if len(i) < 12:
                rates.append(None)
                continue
            d = np.linalg.norm(uv[lag, ids[i]] - uv[lag, ids[j]], axis=1)
            rates.append(float(np.quantile(abs(np.log(np.maximum(d, 1e-3) / d0)), .75) / lag))
        windows.append(dict(start=start, coverage=coverage, pair_log_rate=rates,
                            valid_fraction=[float(valid[k].mean()) if len(q) else 0. for k in [2, 3, 4]]))
    return dict(frame_indices=np.asarray(flow['indices']), affine_thickness=curves, tracklets=windows,
                geometry=mask_geometry(masks[:C + 16]))


def score(sim, real, length):
    """2D loss terms of a simulated drop against the real one."""
    sg, rg = sim['geometry'], real['geometry']
    sc, rc = np.array(sg['center']), np.array(rg['center'])
    ix = np.arange(0, 13)
    pre = float(np.linalg.norm(sc[0:C] - rc[0:C], axis=1).mean())
    center = float(np.linalg.norm(sc[ix] - rc[ix], axis=1).mean() / length)
    velocity = float(np.linalg.norm((sc[ix + 2] - sc[ix]) - (rc[ix + 2] - rc[ix]), axis=1).mean() / length)
    n = min(len(sc), len(rc))
    angle = np.array(sg['angle'])[:n] - np.array(rg['angle'])[:n]
    angle = .5 * abs(np.arctan2(np.sin(2 * angle), np.cos(2 * angle)))
    reliable = (np.array(rg['anisotropy'])[ix] > .18) & (np.array(sg['anisotropy'])[ix] > .18)
    pose = float(np.mean(angle[ix][reliable])) if reliable.sum() >= 3 else None
    rf, sf, tt = np.array(real['affine_thickness']), np.array(sim['affine_thickness']), np.array(real['frame_indices'])
    col = (tt >= C - 1) & (tt <= C + 8)
    affine = float(np.mean(abs(sf[col] - rf[col])) / max(.01, float(np.quantile(rf[col], .9))))
    sx, rx = np.array(sg['extent']), np.array(rg['extent'])
    shape = float(np.mean(abs(np.log((sx[0:C, :2] + 1) / (rx[0:C, :2] + 1)))))
    rates, noise, usable, available = [], [], 0, 0
    for r, s in zip(real['tracklets'], sim['tracklets']):
        if r['start'] < C - 2:
            continue
        for k, (a, b) in enumerate(zip(r['pair_log_rate'], s['pair_log_rate'])):
            if a is None or r['coverage'][k] < .25:
                continue
            available += 1
            noise.append(a)
            if b is not None and s['coverage'][k] >= .25:
                rates.append(abs(b - a))
                usable += 1
    scale = max(.02, float(np.quantile(noise, .9))) if noise else .02
    missing = 1 - usable / max(available, 1)
    short = (float(np.mean(rates) / scale) if rates else 2.) + missing
    return dict(trajectory=center / .1 + .35 * velocity / .05,
                pose=(pose / .35 if pose is not None else 2.) + .5 * shape / .1,
                affine=affine, short_strain=short, preimpact_center_px=pre, missing_short_fraction=missing)
