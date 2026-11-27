"""One PhysReader-Real drop as the replay needs it: scene, initial state, real observations, reference."""
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

C = 3  # contact frame inside the 16-frame window C-3..C+12
FRAMES = 16
FPS = 30.0


def camera_opencv(camera):
    """OpenCV world-to-camera rotation, translation and intrinsics of a camera.json entry."""
    r = np.diag([1., -1., -1.]) @ np.asarray(camera['rotation']).T
    t = -r @ np.asarray(camera['position'])
    k = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1.]])
    return r, t, k


def decode(z, density):
    """(log10 E/rho, log10 sigma_Y/rho) -> E, sigma_Y in Pa."""
    return 10 ** z[0] * density, 10 ** z[1] * density


class Drop:
    """Released files of one drop, e.g. Drop('PhysReader-Real', 'box_1')."""

    def __init__(self, root, key):
        root = Path(root)
        row = next(d for d in json.loads((root / 'drops.json').read_text()) if d['key'] == key)
        d = root / 'drops' / key
        self.key, self.name = key, f"{row['object']}/{row['drop']}"
        self.object, self.density, self.nu = row['object'], row['density'], row['poisson_ratio']
        self.reference = row['reference']
        self.config = json.loads((d / 'replay/replay.json').read_text())
        self.options = self.config['observation']
        self.mesh = trimesh.load(d / 'scene/aligned_mesh.ply', process=False)
        self.camera = json.loads((d / 'scene/camera.json').read_text())[0]
        self.background = cv2.imread(str(d / 'scene/background.png'))
        self.particles = dict(np.load(d / 'scene/particles.npz'))
        self.display = dict(np.load(d / 'replay/display.npz'))
        self.real_masks = np.load(d / 'observations/masks.npz')['masks']
        self.dir = d

    def initial_state(self, eta):
        """Particles and display mesh of the seed state moved by eta = (dx, rotation in deg, dv, d omega)."""
        eta = np.asarray(eta, float)
        p, d = self.particles, self.display
        points = p['positions'].astype(float)
        center = np.average(points, axis=0, weights=p['volumes'])
        rot = Rotation.from_rotvec(np.deg2rad(eta[3:6])).as_matrix()
        positions = ((points - center) @ rot.T + center + eta[:3]).astype(np.float32)
        vertices = (d['vertices'].astype(float) - d['center_before']) @ d['rotation'].T + d['center_after']
        vertices = (vertices - center) @ rot.T + center + eta[:3]
        return positions, p['velocity'] + eta[6:9], p['omega'] + eta[9:12], vertices

    def reference_tracks(self):
        return dict(np.load(self.dir / 'replay/reference_tracks.npz'))


def drops(root):
    return [d['key'] for d in json.loads((Path(root) / 'drops.json').read_text())]
