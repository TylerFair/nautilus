"""Device-independent tests; the same tests can run on a visible GPU."""
import pickle

import h5py
import numpy as np
import pytest

from nautilus import Sampler
from nautilus.bounds import NautilusBound


def identity(x):
    return x


def constant(x):
    return 0.


def test_million_point_contains_sampling_and_volume(tmp_path):
    pytest.importorskip('jax')
    rng = np.random.default_rng(12)
    x = rng.random((600, 3))
    likelihood = -np.linalg.norm(x-.5, axis=1)
    original = NautilusBound.compute(x, likelihood, np.median(likelihood), np.log(.5),
                                     n_networks=1, rng=rng, backend='numpy')
    with h5py.File(tmp_path/'bound.h5', 'w') as f:
        original.write(f)
        a, b = [NautilusBound.read(f, backend=backend, rng=np.random.default_rng(0))
                for backend in ('numpy', 'jax')]
    # Half the million queries bracket a neural outer ellipsoid's surface.
    points = rng.random((1_000_000, 3))
    z = rng.normal(size=(500_000, 3))
    z /= np.linalg.norm(z, axis=1)[:, None]
    z *= np.where(np.arange(len(z)) % 2, 1+1e-6, 1-1e-6)[:, None]
    points[:len(z)] = a.neural_bounds[0].outer_bound.transform(z, inverse=True)
    np.testing.assert_array_equal(a.contains(points), b.contains(points))
    for bound in (a, b):
        bound.reset(np.random.default_rng(1))
        points = bound.sample(20000)
        assert np.all(bound.contains(points))
        assert np.all(a.contains(points)) and np.all(b.contains(points))
        assert bound.n_sample-bound.n_reject == len(points)+len(bound.points)
        if bound.backend == 'jax':
            assert bound.outer_bound.n_sample % 16384 == 0
    def sigma(bound):
        return np.sqrt(sum(v.n_reject/(v.n_sample*(v.n_sample-v.n_reject))
                           for v in (bound, bound.outer_bound)))
    assert abs(a.log_v-b.log_v) < 5*np.hypot(sigma(a), sigma(b))


def test_f_live_caches_not_pickled():
    sampler = Sampler(identity, constant, n_dim=2, backend='numpy')
    sampler._f_live_shell_cache = (1, {1: np.zeros(100)})
    sampler._f_live_key, sampler._f_live_value = (), .5
    restored = pickle.loads(pickle.dumps(sampler))
    assert not any(k.startswith('_f_live_') for k in restored.__dict__)
    assert restored.f_live == 1.
