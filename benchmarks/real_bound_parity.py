"""Check frozen bounds without opening the source checkpoint in HDF5.

The source is copied to /tmp first. All HDF5 access is read-only on that copy.
Use --queries 1000000 --samples 10000 on a GPU for a stronger check.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import h5py
import numpy as np

from nautilus.bounds import NautilusBound


def volume_error(bound):
    return np.sqrt(sum(b.n_reject / max(1, b.n_sample*(b.n_sample-b.n_reject))
                       for b in (bound, bound.outer_bound)))


def check(group, queries, samples, seed):
    bounds = [NautilusBound.read(group, backend=b, rng=np.random.default_rng(seed))
              for b in ('numpy', 'jax')]
    rng = np.random.default_rng(seed)
    points = bounds[0].outer_bound.sample(queries)
    ellipsoid = bounds[0].neural_bounds[0].outer_bound
    z = rng.normal(size=(queries//2, bounds[0].n_dim))
    z /= np.linalg.norm(z, axis=1)[:, None]
    z *= np.where(np.arange(len(z)) % 2, 1+1e-6, 1-1e-6)[:, None]
    points[:len(z)] = ellipsoid.transform(z, inverse=True)
    if bounds[0].shift is not None:
        points = bounds[0].shift.transform(points, inverse=True)
    masks = [b.contains(points) for b in bounds]
    assert np.array_equal(*masks), 'contains parity failed'
    result = dict(queries=queries, inside=int(masks[0].sum()))
    for backend, bound in zip(('numpy', 'jax'), bounds):
        bound.reset(np.random.default_rng(seed))
        bound.sample(1)  # Compile before the steady-state timer.
        bound.reset(np.random.default_rng(seed))
        start = perf_counter()
        points = bound.sample(samples)
        elapsed = perf_counter() - start
        assert all(np.all(b.contains(points)) for b in bounds), 'sample outside bound'
        assert bound.n_sample-bound.n_reject == samples+len(bound.points)
        result[backend] = dict(seconds=elapsed, log_v=float(bound.log_v),
                               sigma=float(volume_error(bound)),
                               raw=bound.outer_bound.n_sample)
    a, b = result['numpy'], result['jax']
    assert abs(a['log_v']-b['log_v']) <= 5*np.hypot(a['sigma'], b['sigma']), 'volume parity failed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--bounds', nargs='+', type=int)
    parser.add_argument('--queries', type=int, default=10000)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nautilus-frozen-', dir='/tmp') as directory:
        copied = Path(directory)/args.checkpoint.name
        shutil.copyfile(args.checkpoint, copied)
        with h5py.File(copied, 'r') as checkpoint:
            indices = args.bounds or sorted(int(k.split('_')[1]) for k in checkpoint
                                             if k.startswith('bound_'))[-3:]
            for index in indices:
                result = check(checkpoint[f'bound_{index}'], args.queries, args.samples, args.seed)
                print(json.dumps(dict(bound=index, **result)), flush=True)
