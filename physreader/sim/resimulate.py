"""Re-run a PhysReader-Sim simulation from its released t=0 state.

    python -m physreader.sim.resimulate --data PhysReader-Sim --sim <id> --out <id>.npz
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np

from .mpm import Solver

FRAME_DT = 0.03333333333
FRICTION = 0.2
WALLS = [
    ((0.05, 0.05, 0.05), (0.0, 1.0, 0.0)),
    ((0.05, 0.05, 0.05), (1.0, 0.0, 0.0)),
    ((0.05, 0.05, 0.05), (0.0, 0.0, 1.0)),
    ((1.95, 1.95, 1.95), (0.0, 0.0, -1.0)),
    ((1.95, 1.95, 1.95), (-1.0, 0.0, 0.0)),
]


def load(data, sim_id):
    """t=0 particle positions and volumes, and the index row of one simulation."""
    data = Path(data)
    row = next(r for r in json.loads((data / 'simulations.json').read_text()) if r['sim'] == sim_id)
    with open(data / row['file'], 'rb') as f:
        f.seek(row['offset'])
        state = np.load(io.BytesIO(f.read(row['size'])))
    return state['positions'], state['volumes'], row


def resimulate(positions, volumes, sim, frames=60, device='cuda:0'):
    """Particle positions at the end of each frame, (frames, N, 3)."""
    solver = Solver(positions, volumes, sim['E'], sim['nu'], sim['density'], sim['yield_stress'],
                    velocity=sim['initial_velocity'], repaired=sim['solver'] == 'repaired', device=device)
    # the data generator registered every wall twice
    for _ in range(2):
        for point, normal in WALLS:
            solver.add_wall(point, normal, FRICTION)
    dt = float(sim['substep_dt'])
    track = np.empty((frames, len(positions), 3), np.float32)
    for frame in range(frames):
        for _ in range(int(FRAME_DT / dt)):
            solver.step(dt)
        track[frame] = solver.positions()
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='PhysReader-Sim folder')
    ap.add_argument('--sim', required=True, help='simulation id, see clip_to_simulation.json')
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    positions, volumes, sim = load(a.data, a.sim)
    np.savez_compressed(a.out, positions=resimulate(positions, volumes, sim, a.frames))


if __name__ == '__main__':
    main()
