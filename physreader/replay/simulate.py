"""Simulate a real drop over the window C-3..C+12 and render it over the drop's own background."""
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
import warp as wp

from ..sim.mpm import Solver
from .data import FPS, FRAMES, camera_opencv

FRICTION = 0.2
WALLS = [
    ((0.05, 0.05, 0.05), (0.0, 1.0, 0.0)),
    ((0.05, 0.05, 0.05), (1.0, 0.0, 0.0)),
    ((0.05, 0.05, 0.05), (0.0, 0.0, 1.0)),
    ((1.95, 1.95, 1.95), (0.0, 0.0, -1.0)),
    ((1.95, 1.95, 1.95), (-1.0, 0.0, 0.0)),
]
N_GRID, GRID_LIM = 80, 2.0
DX = GRID_LIM / N_GRID


def substep(E, nu, density):
    """Largest dt below 2e-4 s and 0.3 dx / (P-wave speed) that divides one frame.
    E passes through units of 1e7 Pa, as in the scene configuration."""
    E = E / 1e7 * 1e7
    speed = math.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * density))
    steps = math.ceil(1 / FPS / min(2e-4, .3 * DX / speed))
    return 1 / FPS / steps, steps


class Renderer:
    """Unlit vertex colors, one face per pixel (PyTorch3D)."""

    def __init__(self, vertices, faces, colors, camera, device='cuda:0'):
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesVertex
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        r, t, k = camera_opencv(camera)
        size = [[camera['height'], camera['width']]]
        cameras = cameras_from_opencv_projection(tensor(r[None]), tensor(t[None]), tensor(k[None]), tensor(size))
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
            image_size=tuple(size[0]), blur_radius=0, faces_per_pixel=1, perspective_correct=True,
            cull_backfaces=False, max_faces_per_bin=200000))
        self.mesh = Meshes(verts=[tensor(vertices)], faces=[torch.as_tensor(faces, device=device, dtype=torch.int64)],
                           textures=TexturesVertex(verts_features=[tensor(colors[:, :3] / 255.)]))

    @torch.no_grad()
    def __call__(self, vertices):
        mesh = self.mesh.update_padded(vertices[None])
        fragments = self.rasterizer(mesh)
        mask = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()
        rgb = (mesh.sample_textures(fragments)[0, ..., 0, :].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        depth = fragments.zbuf[0, ..., 0].cpu().numpy()
        depth[~mask] = 0
        return rgb, mask, depth


def ffmpeg():
    exe = os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_h264(src, dst):
    subprocess.run([ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y', '-threads', '2', '-i', str(src),
                    '-map', '0:v:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst)], check=True)


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_solver(drop, E, yield_stress, positions, velocity, omega, device='cuda:0'):
    """Repaired solver for the drop's scene, every particle moving with velocity + omega x (x - center of mass)."""
    x = torch.tensor(positions, device=device)
    vol = torch.tensor(drop.particles['volumes'], device=device)
    center = (x * vol[:, None]).sum(0) / vol.sum()
    v = torch.tensor(velocity, dtype=x.dtype, device=device) + torch.cross(
        torch.tensor(omega, dtype=x.dtype, device=device).expand_as(x), x - center, dim=1)
    solver = Solver(positions, drop.particles['volumes'], E, drop.nu, drop.density, yield_stress,
                    velocity=v.cpu().numpy(), n_grid=N_GRID, grid_lim=GRID_LIM, repaired=True, device=device)
    for point, normal in WALLS:
        solver.add_wall(point, normal, FRICTION)
    return solver


def simulate(drop, E, yield_stress, eta, video=None, device='cuda:0'):
    """Particle positions (16, N, 3) from the drop's seed state moved by eta; E, yield stress in Pa.
    With `video` (an .mp4 path) the drop is also rendered, see render()."""
    positions, velocity, omega, vertices = drop.initial_state(eta)
    solver = make_solver(drop, E, yield_stress, positions, velocity, omega, device)
    dt, steps = substep(E, drop.nu, drop.density)
    lo, hi = DX * .5, GRID_LIM - 1.5 * DX
    x, v = wp.to_torch(solver.particles.x), wp.to_torch(solver.particles.v)
    track = np.empty((FRAMES,) + positions.shape, np.float32)
    for frame in range(FRAMES):
        for _ in range(steps if frame else 0):
            solver.step(dt)
            if not bool(torch.isfinite(x).all() & torch.isfinite(v).all()):
                raise FloatingPointError(f'nonfinite MPM state in frame {frame}')
            if bool((x < lo).any() | (x >= hi).any()):
                raise FloatingPointError(f'MPM particles left the safe grid in frame {frame}')
        track[frame] = solver.positions()
    out = dict(positions=track, initial=positions, vertices=vertices.astype(np.float32))
    if video is not None:
        out.update(render(drop, track, vertices, video, device))
    return out


def render(drop, track, vertices, video, device='cuda:0'):
    """The display mesh (vertices at frame 0) follows the surface particles of `track`; rendered over the
    background and stored as H.264 like the real videos. Returns the decoded RGB frames, masks and depth."""
    renderer = Renderer(vertices, drop.mesh.faces, np.asarray(drop.mesh.visual.vertex_colors), drop.camera, device)
    rest = torch.tensor(vertices, dtype=torch.float32, device=device)
    n = int(drop.particles['num_surface'])
    controls = torch.tensor(track[0, :n], device=device)
    nb = torch.tensor(drop.display['neighbors'], dtype=torch.long, device=device)
    wt = torch.tensor(drop.display['weights'], device=device)
    h, w = drop.background.shape[:2]
    raw = str(video) + '.mp4v.mp4'
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    masks, depths = [], []
    for x in track:
        rgb, mask, depth = renderer(rest + ((torch.tensor(x[:n], device=device) - controls)[nb] * wt[..., None]).sum(1))
        frame = drop.background.copy()
        frame[mask] = rgb[..., ::-1][mask]
        writer.write(frame)
        masks.append(mask)
        depths.append(depth)
    writer.release()
    to_h264(raw, video)
    os.remove(raw)
    return dict(masks=np.stack(masks), depth=np.stack(depths), frames=read_video(video))
