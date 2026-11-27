"""Initial-condition search with the material held fixed: 25 trials, a Gaussian-process expected-improvement
search around the best guarded trial, plus paired random steps."""
import hashlib

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

WEIGHTS = dict(trajectory=.25, pose=.25, affine=.45, short_strain=.05)
PRIOR = .02
# eta = (position m, rotation deg, velocity m/s, angular velocity rad/s), three components each
SIG = np.array([.015] * 3 + [8.] * 3 + [.2] * 3 + [.7] * 3)
BOUNDS = np.array([.03] * 3 + [10.] * 3 + [.35] * 3 + [.8] * 3)
RADIUS = np.array([.005] * 3 + [2.] * 3 + [.06] * 3 + [.2] * 3)


def total(m):
    """Replay loss reported in the paper (no IC prior)."""
    return sum(w * m[k] for k, w in WEIGHTS.items())


def deformation(m):
    return .9 * m['affine'] + .1 * m['short_strain']


def prior(eta):
    return float(np.mean((np.asarray(eta) / SIG) ** 2))


def scalar(row):
    return total(row['metrics']) + PRIOR * row['metrics']['prior']


def _slack(anchor, tolerance, k):
    return max(.05 * max(anchor[k], .05), 2 * tolerance[k], .01)


def guard(row, anchor, tolerance):
    """Trajectory and pose not much worse than the reference's, and similar tracklet support."""
    m = row['metrics']
    return bool(all(m[k] <= anchor[k] + _slack(anchor, tolerance, k) for k in ['trajectory', 'pose'])
                and m['preimpact_center_px'] <= max(8., anchor['preimpact_center_px'] * 1.05 + 1.)
                and m['missing_short_fraction'] <= min(1., anchor['missing_short_fraction'] + .10))


def penalty(row, anchor, tolerance):
    v = scalar(row)
    for k in ['trajectory', 'pose']:
        v += 2 * max(0, row['metrics'][k] - anchor[k] - _slack(anchor, tolerance, k))
    return v + .03 * max(0, row['metrics']['preimpact_center_px'] - max(8., anchor['preimpact_center_px'] * 1.05 + 1.))


def proposals(history, anchor, tolerance, cycle, seed):
    """Six paired random steps around the best trial and two expected-improvement picks."""
    rng = np.random.default_rng(seed + cycle)
    best = min(history, key=lambda r: (not guard(r, anchor, tolerance), penalty(r, anchor, tolerance)))
    eta = np.array(best['initial_state'])
    rad, etas = RADIUS * .85 ** cycle, []
    for _ in range(3):
        v = rng.normal(size=12)
        v /= max(abs(v).max(), 1e-8)
        etas += [np.clip(eta + sign * rad * v, -BOUNDS, BOUNDS) for sign in [-1, 1]]
    pool = np.clip(eta + rng.uniform(-1.5, 1.5, (2500, 12)) * rad, -BOUNDS, BOUNDS)
    xx = np.array([r['initial_state'] for r in history]) / SIG
    yy = np.array([penalty(r, anchor, tolerance) for r in history])
    gp = GaussianProcessRegressor(kernel=Matern(np.ones(12), nu=2.5), alpha=1e-4, normalize_y=True, optimizer=None)
    gp.fit(xx, yy)
    mu, sd = gp.predict(pool / SIG, return_std=True)
    diff = min(yy) - mu - .002
    z = diff / np.maximum(sd, 1e-9)
    ei = diff * norm.cdf(z) + sd * norm.pdf(z)
    pick = []
    for k in np.argsort(-ei):
        if all(np.linalg.norm((pool[k] - e) / SIG) > .15 for e in pick):
            pick.append(pool[k])
        if len(pick) == 2:
            break
    return etas + pick


def search(evaluate, name, start, anchor, tolerance, budget=25, log=print):
    """evaluate(i, eta) -> loss terms of trial i. Returns the guarded trial with the lowest loss plus prior
    (the lowest overall if no trial passes the guard), all trials and the failed ones."""
    seed = 1701 + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    plan, history, failures = [list(start)], [], []
    for i in range(budget):
        if i == len(plan):
            if history:
                props = proposals(history, anchor, tolerance, (i - 1) // 8, seed)
            else:
                rng = np.random.default_rng(seed + i)
                props = [np.clip(rng.normal(size=12) * RADIUS, -BOUNDS, BOUNDS) for _ in range(8)]
            plan += [x.tolist() for x in props]
        try:
            metrics = dict(evaluate(i, plan[i]), prior=prior(plan[i]))
            history.append(dict(trial=i, initial_state=plan[i], metrics=metrics))
            log(f'{name} trial {i}: loss {scalar(history[-1]):.4f}')
        except Exception as e:
            failures.append(dict(trial=i, initial_state=plan[i], error=repr(e)))
            log(f'{name} trial {i} failed: {e!r}')
    if not history:
        raise RuntimeError(f'{name}: all {budget} trials failed')
    eligible = [r for r in history if guard(r, anchor, tolerance)]
    return dict(selected=min(eligible or history, key=scalar), guard_passed=bool(eligible), trials=history,
                failures=failures)
