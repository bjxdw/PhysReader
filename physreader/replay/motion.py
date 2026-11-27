"""3D motion of the visible surface over short windows (lags of 2 and 3 frames): robust rigid fits of
visible simulated surface points, compared with the real drop's depth-lifted flow tracklets."""
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .data import camera_opencv
from .observe import sample, window_starts


class Camera:
    def __init__(self, camera):
        self.R, self.T, self.K = camera_opencv(camera)

    def lift(self, uv, z):
        K = self.K
        cam = np.stack([(uv[..., 0] - K[0, 2]) * z / K[0, 0], (uv[..., 1] - K[1, 2]) * z / K[1, 1], z], -1)
        return (cam - self.T) @ self.R

    def project(self, p):
        cam = p @ self.R.T + self.T
        return cam[..., :2] / cam[..., 2, None] * np.array([self.K[0, 0], self.K[1, 1]]) + self.K[:2, 2], cam[..., 2]


def ray_bind(camera, vertices, faces, uv):
    """First triangle hit by each pixel ray and the barycentric coordinates of the hit."""
    face, bary = np.full(len(uv), -1, int), np.zeros((len(uv), 3))
    if len(uv) == 0:
        return face, bary
    center = -camera.T @ camera.R
    loc, ray, tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).ray.intersects_location(
        np.broadcast_to(center, (len(uv), 3)), camera.lift(uv, np.ones(len(uv))) - center, multiple_hits=False)
    if len(ray):
        face[ray] = tri
        bary[ray] = trimesh.triangles.points_to_barycentric(vertices[faces[tri]], loc)
    return face, bary


def rigid(a, b):
    weights, rot = np.ones(len(a)), np.eye(3)
    for _ in range(5):
        w = weights / max(weights.sum(), 1e-9)
        ca, cb = (a * w[:, None]).sum(0), (b * w[:, None]).sum(0)
        U, sv, Vh = np.linalg.svd(((a - ca) * w[:, None]).T @ (b - cb))
        rot = Vh.T @ np.diag([1, 1, np.linalg.det(Vh.T @ U.T)]) @ U.T
        e = np.linalg.norm((a - ca) @ rot.T + cb - b, axis=1)
        scale = max(.002, float(np.median(e)) * 1.4826)
        weights = np.minimum(1, 1.5 * scale / np.maximum(e, 1e-8))
    return rot, np.median(b - a, axis=0), sv[1] / max(sv[0], 1e-12)


def measurement(a, b, start, lag, bootstrap=24):
    """Rigid motion of paired points a -> b with bootstrap uncertainty."""
    if len(a) < 10:
        return dict(start=start, lag=lag, valid=False)
    R, d, condition = rigid(a, b)
    rng, moves, angles = np.random.default_rng(100 + start * 10 + lag), [], []
    for _ in range(bootstrap):
        ix = rng.integers(0, len(a), len(a))
        q, delta, _ = rigid(a[ix], b[ix])
        moves.append(delta)
        angles.append(Rotation.from_matrix(q @ R.T).magnitude())
    spread = np.quantile(np.linalg.norm(np.array(moves) - d, axis=1), .9)
    return dict(start=start, lag=lag, valid=True, velocity_m_s=d / (lag / 30), rotation=R,
                rotation_bootstrap_rad=float(np.quantile(angles, .9)), motion_uncertainty_m=max(float(spread), .002),
                rotation_reliable=bool(condition > .025 and np.quantile(angles, .9) < np.deg2rad(12)))


def surface_motion(drop, sim):
    """Windows of visible simulated surface points: seeded on the rendered mask, bound to the mesh by ray
    casting, kept where still visible (mask and depth) after the lag."""
    camera, faces, o = Camera(drop.camera), np.asarray(drop.mesh.faces), drop.options
    rest, nb, wt = sim['vertices'].astype(float), drop.display['neighbors'].astype(np.int64), drop.display['weights']
    tracks, initial, masks, depth = sim['positions'], sim['initial'], sim['masks'], sim['depth']
    cache = {}

    def vertices(t):
        if t not in cache:
            cache[t] = rest + np.sum((tracks[t, nb] - initial[nb]) * wt[:, :, None], axis=1)
        return cache[t]

    rows, erode, spacing = [], o['surface_erode'], o['surface_spacing']
    for start in window_starts(o['window_stride']):
        y, x = np.where(cv2.erode(masks[start].astype('uint8'), np.ones((erode, erode), np.uint8)))
        keep = (x % spacing == 0) & (y % spacing == 0)
        uv = np.c_[x[keep], y[keep]].astype(float)
        if len(uv) > 140:
            uv = uv[np.linspace(0, len(uv) - 1, 140).astype(int)]
        face, bary = ray_bind(camera, vertices(start), faces, uv)
        face, bary = face[face >= 0], bary[face >= 0]
        a = (vertices(start)[faces[face]] * bary[:, :, None]).sum(1)
        for lag in [2, 3]:
            b = (vertices(start + lag)[faces[face]] * bary[:, :, None]).sum(1)
            q, z = camera.project(b)
            vis = ((q[:, 0] >= 1) & (q[:, 0] < masks.shape[2] - 1) & (q[:, 1] >= 1) & (q[:, 1] < masks.shape[1] - 1)
                   & (sample(masks[start + lag], q) > .99) & (abs(sample(depth[start + lag], q) - z) < .012))
            rows.append(measurement(a[vis], b[vis], start, lag))
    return rows


def silhouette_distance(sim_masks, real_masks):
    """Mean symmetric boundary distance (px, capped at 40) after removing the centroid offset, frames 0..12."""
    errs = []
    for t in range(0, 13):
        a, b = sim_masks[t].astype('uint8'), real_masks[t].astype('uint8')
        ca, cb = np.argwhere(a)[:, ::-1].mean(0), np.argwhere(b)[:, ::-1].mean(0)
        a = cv2.warpAffine(a, np.array([[1, 0, cb[0] - ca[0]], [0, 1, cb[1] - ca[1]]], dtype='float32'),
                           (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST)
        ea = a.astype(bool) & ~cv2.erode(a, np.ones((3, 3), np.uint8)).astype(bool)
        eb = b.astype(bool) & ~cv2.erode(b, np.ones((3, 3), np.uint8)).astype(bool)
        da = cv2.distanceTransform((~ea).astype('uint8'), cv2.DIST_L2, 5)
        db = cv2.distanceTransform((~eb).astype('uint8'), cv2.DIST_L2, 5)
        errs.append(float(.5 * (np.minimum(da[eb], 40).mean() + np.minimum(db[ea], 40).mean())))
    return float(np.mean(errs))


def score(real, rows, sim_masks, real_masks):
    """3D trajectory and pose terms: velocity direction/vertical/horizontal trends and relative rotations
    over the windows where the real drop has reliable motion, plus the silhouette boundary distance."""
    pairs = []
    for a, b in zip(real['rows'], rows):
        assert (a['start'], a['lag']) == (b['start'], b['lag'])
        if a['valid']:
            pairs.append((a, b))
    scale = max(.25, float(np.quantile([np.linalg.norm(a['velocity_m_s']) for a, b in pairs], .8)))
    terms, pose = [], []
    for a, b in pairs:
        if not b['valid']:
            terms.append([2, 2, 2])
            continue
        va, vb = np.array(a['velocity_m_s']), np.array(b['velocity_m_s'])
        noise = max(.04, a['motion_uncertainty_m'] / (a['lag'] / 30), min(.4, real['background_depth_jitter_m'] * 30))
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        direction = (1 - float(np.dot(va, vb) / (na * nb))) if na > noise and nb > noise else (1. if na > noise else 0.)
        vertical = abs(va[1] - vb[1]) / scale
        horizontal = abs(np.linalg.norm(va[[0, 2]]) - np.linalg.norm(vb[[0, 2]])) / scale
        terms.append([direction, vertical, horizontal])
        if a['rotation_reliable'] and b['rotation_reliable']:
            error = Rotation.from_matrix(np.array(b['rotation']) @ np.array(a['rotation']).T).magnitude()
            pose.append(float(error) / max(a['rotation_bootstrap_rad'], np.deg2rad(8)))
    if len(terms) < 5:
        raise ValueError('fewer than 5 reliable real 3D motion windows')
    d, v, h = np.mean(terms, axis=0)
    shape = silhouette_distance(sim_masks, real_masks)
    return dict(trajectory=float(.55 * d + .30 * v + .15 * h),
                pose=float(.6 * np.mean(pose) + .4 * shape / 10) if len(pose) >= 3 else float(shape / 10 + 1))
