"""Err_ref: a material simulated from the reference's initial state, compared with the reference simulation
particle by particle (center trajectory, rotation, point-track-fitted Hencky strain)."""
import math

import numpy as np
from scipy.spatial import cKDTree


def rotation(x, y, w):
    a, b = x - np.average(x, axis=0, weights=w), y - np.average(y, axis=0, weights=w)
    u, s, vt = np.linalg.svd((a * w[:, None]).T @ b)
    d = np.ones(3)
    d[-1] = np.linalg.det(u @ vt)
    return (u * d) @ vt


def strain(x, ids, coef):
    """Hencky strain of the least-squares F over each particle's 24 neighbors, and where it is valid."""
    F = np.einsum('nki,nkj->nij', x[ids] - x[:, None, :], coef)
    ev, Q = np.linalg.eigh(np.swapaxes(F, 1, 2) @ F)
    valid = (ev[:, 0] > 1e-12) & np.isfinite(ev).all(1) & (np.linalg.det(F) > 0)
    return (Q * (.5 * np.log(np.maximum(ev, 1e-12)))[:, None, :]) @ np.swapaxes(Q, 1, 2), valid


def err_ref(x, reference):
    """x: (16, N, 3) prediction tracks; reference: reference_tracks.npz of the drop."""
    x, y, x0 = np.asarray(x, float), reference['positions'].astype(float), reference['start'].astype(float)
    w, rest = reference['volumes'].astype(float), reference['rest'].astype(float)
    assert x.shape == y.shape and np.isfinite(x).all()
    L = float(np.linalg.norm(np.ptp(rest, axis=0)))
    cx, cy = np.einsum('tnc,n->tc', x, w) / w.sum(), np.einsum('tnc,n->tc', y, w) / w.sum()
    trajectory = float(np.sqrt(np.mean(np.sum((cx - cy) ** 2, axis=1))) / L)
    angles = []
    for a, b in zip(x, y):
        ra, rb = rotation(rest, a, w), rotation(rest, b, w)
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1) / 2, -1, 1)))))
    pose = float(np.mean(angles))
    ids = cKDTree(x0).query(x0, k=25, workers=1)[1][:, 1:]
    a = x0[ids] - x0[:, None, :]
    ev, Q = np.linalg.eigh(np.einsum('nki,nkj->nij', a, a))
    base = (ev[:, 0] > 1e-6 * np.maximum(ev[:, -1], 1e-30)) & (ev[:, -1] > 1e-20)
    coef = a @ ((Q * (1 / np.maximum(ev, 1e-30))[:, None, :]) @ np.swapaxes(Q, 1, 2))
    numerator = denominator = 0.
    for a, b in zip(x[2:12], y[2:12]):
        ha, va = strain(a, ids, coef)
        hb, vb = strain(b, ids, coef)
        ok = base & va & vb
        numerator += float(w[ok] @ np.sum((ha - hb) ** 2, axis=(1, 2))[ok])
        denominator += float(w[ok].sum())
    H = math.sqrt(numerator / denominator)
    return dict(trajectory=trajectory, pose_deg=pose, point_fit_strain_rmse=H,
                total_with_point_fit_strain=.25 * trajectory + .25 * pose / 180 + .5 * H)
