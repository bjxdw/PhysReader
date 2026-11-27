# What a Drop Reveals: Sim-to-Real Material Estimation for Deformable Objects from Video

[Project page](https://bjxdw.github.io/PhysReader/) · [Paper](https://arxiv.org/abs/2611.34894) ·
[PhysReader-Real](https://huggingface.co/datasets/bjxdw/PhysReader-Real) ·
[PhysReader-Sim](https://huggingface.co/datasets/bjxdw/PhysReader-Sim) ·
[Models](https://huggingface.co/bjxdw/PhysReader)

PhysReader estimates the density-normalized stiffness E/ρ and yield stress σ<sub>Y</sub>/ρ of a deformable
object from one video of it dropping. It is trained only on simulated drops and evaluated on PhysReader-Real:
16 everyday objects, each dropped five times.

## Setup

```bash
git clone --single-branch https://github.com/bjxdw/PhysReader && cd PhysReader
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

We ran everything on Linux with NVIDIA A100 GPUs. The V-JEPA 2 backbone (`facebook/vjepa2-vitl-fpc64-256`)
is downloaded from Hugging Face the first time a model is built. The simulator needs NVIDIA Warp 0.10.1,
the version that generated the data.

## Data and models

| | Size | Contents |
|---|---|---|
| [PhysReader-Real](https://huggingface.co/datasets/bjxdw/PhysReader-Real) | 15 GB | 80 real drops: video, LiDAR depth, aligned scene, observations, model inputs, reference materials, plausible regions, tensile tests |
| [PhysReader-Sim](https://huggingface.co/datasets/bjxdw/PhysReader-Sim) | 180 GB | 291,638 simulated clips (277,050 train / 14,588 val) from 97,226 simulations: model inputs, labels, source videos, teacher inputs and latents, t=0 simulation states |
| [PhysReader](https://huggingface.co/bjxdw/PhysReader) | 171 MB | the teacher and every Table 1 model |

```bash
hf download bjxdw/PhysReader --local-dir checkpoints
hf download bjxdw/PhysReader-Real --repo-type dataset --local-dir data/PhysReader-Real
hf download bjxdw/PhysReader-Sim --repo-type dataset --local-dir data/PhysReader-Sim   # or --include "val*" "*/val-*"
```

## Reproducing the paper

### Table 1, real drops

```bash
python -m physreader.predict --model physreader --checkpoint checkpoints/physreader.pt \
    --inputs data/PhysReader-Real/inputs --real data/PhysReader-Real --out real.json
```

prints the mean parameter distance over the 80 drops. `--model` selects a row:

| Input | `--model` | Syn. | Parameter distance |
|---|---|---|---|
| RGB | `rgb` | 0.478 | 0.797 |
| RGB + dense flow | `rgb_flow` | 0.472 | 0.833 |
| RGB + depth | `rgb_depth` | 0.477 | 0.789 |
| 2D tracklets only | `tracks_2d` | 0.764 | 1.080 |
| 3D tracklets only | `tracks_3d` | 0.702 | 1.337 |
| RGB + 2D tracklets | `rgb_tracks_2d` | 0.482 | 0.745 |
| RGB + 2D tracklets + cues | `rgb_tracks_2d_cues` | 0.487 | 0.739 |
| RGB + 3D tracklets | `rgb_tracks_3d` | 0.444 | 1.210 |
| + depth augmentation | `rgb_tracks_3d_depth_aug` | 0.455 | 0.928 |
| Student, 3D tracklets | `student_3d` | 0.443 | 1.164* |
| PhysReader | `physreader` | 0.481 | 0.677 |

\* 1.1649, printed as 1.165; the paper rounds it down.

### Table 1, replay columns

```bash
git clone https://github.com/princeton-vl/SEA-RAFT   # with Tartan-C-T-TSKH-spring540x960-M.pth in SEA-RAFT/models
python -m physreader.replay --data data/PhysReader-Real --predictions real.json --sea-raft SEA-RAFT --out replay/physreader
```

simulates every drop over the window C−3..C+12 with the predicted material held fixed (decoded with the drop's
density, never clipped), tries 25 initial conditions and compares the best one with the reference's replay loss (Δ
total and Δ deformation, in percent). It also reports Err_ref: the material simulated from the reference's initial
state and compared particle by particle with the reference simulation. `--drops box_1 ...` selects drops,
`--skip-replay` computes Err_ref only, and `--reference --fixed` replays the reference materials at their selected
initial states. For PhysReader the paper reports Δ deformation 12.32%, Δ total 10.15% and Err_ref 0.1072; a re-run
lands within about one point of the Δ values (see below). Results are written per drop, so an interrupted run
resumes. On one A100 a drop takes about 4 minutes for a soft material and up to 40 for the stiffest; PhysReader's
80 drops take about 8 hours.
The replay also needs PyTorch3D 0.7.9, trimesh 4.4 with embreex, scikit-learn and imageio-ffmpeg.

### Synthetic validation

```bash
python -m physreader.predict --model physreader --checkpoint checkpoints/physreader.pt \
    --inputs data/PhysReader-Sim/val.json --out val.json
```

prints the stiffness and yield-stress MAE (dex). Sim clips store no RGB: each 16-frame window is cut from the
source video with the crop used in training, so a pass over the 14,588 validation clips spends most of its
time decoding video. The teacher (`--model teacher`, 0.161) reads `val.json` the same way.

### Training

Every model trains with a global batch of 480 clips at any GPU count (4 clips per GPU, gradient accumulation
makes up the rest). We used 6 to 32 A100 40 GB GPUs per run.

```bash
# direct models (any row except the students)
torchrun --nproc_per_node 8 -m physreader.train --model rgb_tracks_2d --data data/PhysReader-Sim --out runs/rgb_tracks_2d

# PhysReader: teacher, its latents, then the student
torchrun --nproc_per_node 8 -m physreader.train --model teacher --data data/PhysReader-Sim --out runs/teacher
python -m physreader.export_latents --teacher runs/teacher/best.pt --data data/PhysReader-Sim --out runs/latents
torchrun --nproc_per_node 8 -m physreader.train --model physreader --data data/PhysReader-Sim --out runs/physreader \
    --teacher runs/teacher/best.pt --latents runs/latents/train.npz
```

With the released teacher (`checkpoints/teacher.pt`) and its latents (`data/PhysReader-Sim/latents/train.npz`)
you can skip the first two steps. The teacher trains
for 10 epochs and keeps the best validation epoch; every other model trains for 5 epochs and the paper reports
the last one. `rgb_flow`, `rgb_depth` and `rgb_tracks_3d_depth_aug` need dense flow or depth for every training
clip, which we did not release; they can be evaluated but not retrained from the release.

### Re-running a simulation

Every clip comes from a simulation whose t=0 state (particle positions and volumes), material and initial
velocity are in `data/PhysReader-Sim/simulations`:

```bash
python -m physreader.sim.resimulate --data data/PhysReader-Sim --sim <id> --out sim.npz
```

`clip_to_simulation.json` maps a clip to its simulation. The re-run matches the released trajectories to about
0.01 mm. Where the contact is chaotic, runs can drift apart by up to a few mm, because GPU atomic sums are not
reproducible from run to run; the original code behaves the same. 70 of the 97,226 simulations (210 clips) cannot
be re-run exactly: their particle volumes could not be recovered, marked by `abs(mass_check - 1) > 1e-3` in
`simulations.json`.

## Differences from the paper

- The released clips are the ones we trained on, but the rendering cannot be regenerated: the mesh scale, texture
  and particle sampling seeds were not recorded. We release the videos instead.
- Replays are not bitwise reproducible. GPU atomic sums move particles by a few µm from run to run (0.1 mm for
  the stiffest objects), and the deformation terms, computed from optical flow on the rendered video, amplify
  this: re-running one trial changes its total loss by about 1%, its deformation loss by several percent, more
  for objects with few flow tracklets (belt) or very stiff ones (box). The original code behaves the same.
  Err_ref reproduces the paper to 1e-6; for the stiffest drops it changes by a few 1e-3 from run to run.
- For seven rows of Table 1 (RGB, RGB + dense flow, RGB + depth, 2D tracklets only, RGB + 2D tracklets,
  RGB + 2D tracklets + cues, RGB + 3D tracklets), the replay columns take 43–46 of the 80 drops from an earlier
  version of the replay that simulated 60 frames and weighted the loss terms differently. This code implements the
  16-frame protocol used for the other drops, for both students and for the references, so re-running those rows
  gives somewhat different replay numbers. Their parameter distances are not affected.

## Citation

```bibtex
@article{peng2026physreader,
  title   = {What a Drop Reveals: Sim-to-Real Material Estimation for Deformable Objects from Video},
  author  = {Ziyue Peng and Yanxi Hu and Junyoung Kim and Andrew Chen and Jin Yang and Chandrajit L. Bajaj and Etienne Vouga and William T. Freeman and Qixing Huang and Brandon Y. Feng},
  journal = {arXiv preprint arXiv:2611.34894},
  year    = {2026}
}
```

## License

Code: MIT. Data and model weights: see the dataset and model cards.
