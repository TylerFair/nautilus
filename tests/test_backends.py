"""Backend dispatch, geometry, checkpoint interchange, and native fixes."""
import builtins
from copy import deepcopy

import h5py
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import kstest

from nautilus import Sampler
from nautilus.bounds import NautilusBound, Union, UnitCubeEllipsoidMixture
from nautilus.bounds.basic import minimum_volume_enclosing_ellipsoid
from nautilus.neural import NeuralNetworkEmulator, resolve_backend


def test_backend_dispatch(monkeypatch):
    assert resolve_backend('jax', pool=object()) == 'numpy'
    with pytest.raises(ValueError):
        resolve_backend('invalid')
    original = builtins.__import__

    def without_jax(name, *args, **kwargs):
        if name == 'jax':
            raise ImportError('JAX intentionally absent')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_jax)
    assert resolve_backend('auto') == resolve_backend('numpy') == 'numpy'
    with pytest.raises(ImportError, match='requires JAX'):
        resolve_backend('jax')


def test_gpu_default(monkeypatch):
    jax = pytest.importorskip('jax')
    from types import SimpleNamespace
    for platform, expected in [('cpu', 'numpy'), ('gpu', 'jax')]:
        monkeypatch.setattr(jax, 'devices', lambda: [SimpleNamespace(platform=platform)])
        assert resolve_backend('auto') == expected


@pytest.mark.parametrize('source,target', [('numpy', 'jax'), ('jax', 'numpy')])
def test_checkpoint_backend_interchange(tmp_path, source, target):
    pytest.importorskip('jax')
    path = tmp_path / 'sampler.h5'
    kwargs = dict(n_dim=2, n_live=100, n_networks=1, seed=7, filepath=path)
    prior = lambda x: x
    likelihood = lambda x: -np.sum((x - .5)**2) / .03
    original = Sampler(prior, likelihood, backend=source, **kwargs)
    original.run(n_like_max=1500, n_eff=2000)
    assert len(original.bounds) > 1
    resumed = Sampler(prior, likelihood, backend=target, **kwargs)
    assert resumed.log_z == original.log_z
    for a, b in zip(original.posterior(), resumed.posterior()):
        np.testing.assert_array_equal(a, b)
    with h5py.File(path) as f:
        assert 'backend' not in f['sampler'].attrs
    emulator = resumed.bounds[-1].neural_bounds[0].emulator
    x = np.random.default_rng(0).normal(size=(1025, 2))
    stock = np.mean([n.predict((x - emulator.mean) / emulator.scale)
                     for n in emulator.neural_networks], axis=0)
    np.testing.assert_allclose(emulator.predict(x), stock, atol=1e-12, rtol=1e-12)
    resumed.run(n_like_max=resumed.n_like + 300, n_eff=3000)
    assert np.isfinite(resumed.log_z)


def test_device_geometry_and_rng():
    jax = pytest.importorskip('jax')
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 3)) * [.06, .03, .001] + .5
    bound = Union.compute(x, bound_class=UnitCubeEllipsoidMixture, rng=rng, backend='jax')
    query = rng.uniform(size=(20000, 3))
    saved = deepcopy(rng.bit_generator.state)
    mask = bound.contains(query)
    assert rng.bit_generator.state == saved
    bound.backend = 'numpy'
    np.testing.assert_array_equal(mask, bound.contains(query))
    bound.backend = 'jax'
    points = bound.sample(20000)
    assert np.all(bound.contains(points))
    assert bound.n_sample % 16384 == 0
    assert bound.n_sample - bound.n_reject == len(points) + len(bound.points)
    # A symmetric ellipsoid has uniform marginal CDF after whitening.
    ellipsoid = bound.bounds[0].ellipsoid
    z = ellipsoid.transform(points[:, ~bound.bounds[0].dim_cube])[:, 0]
    assert kstest(z, lambda z: .5 + .75*z - .25*z**3).pvalue > .001
    bound.reset()
    rng.bit_generator.state = saved
    np.testing.assert_array_equal(points, bound.sample(20000))
    before = jax.config.jax_enable_x64
    c, a, inverse = minimum_volume_enclosing_ellipsoid(x, backend='jax')
    assert jax.config.jax_enable_x64 == before
    assert np.max(np.einsum('ni,ij,nj->n', x-c, a, x-c)) <= 1 + 1e-8
    np.testing.assert_allclose(a @ inverse, np.eye(3), atol=1e-7)


def test_jax_training_and_prediction(tmp_path):
    jax = pytest.importorskip('jax')
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4096, 3))
    y = np.linalg.norm(x, axis=1) / 5
    kwargs = dict(n_networks=2, neural_network_kwargs={'max_iter':80}, backend='jax')
    before = jax.config.jax_enable_x64
    emulator = NeuralNetworkEmulator.train(x, y, **kwargs)
    assert np.mean((emulator.predict(x) - y)**2) < .02
    assert jax.config.jax_enable_x64 == before
    with h5py.File(tmp_path / 'networks.h5', 'w') as group:
        emulator.write(group)
        stock = NeuralNetworkEmulator.read(group, backend='numpy')
        np.testing.assert_allclose(emulator.predict(x), stock.predict(x), atol=1e-12)
    again = NeuralNetworkEmulator.train(x, y, **kwargs)
    for a, b in zip(emulator.neural_networks, again.neural_networks):
        np.testing.assert_array_equal(a.coefs_[0], b.coefs_[0])
        assert a.n_iter_ <= 80


def test_27d_conditioned_surface():
    pytest.importorskip('jax')
    rng = np.random.default_rng(5)
    bound = Union.compute(rng.normal(size=(100, 27)), unit=False, backend='numpy')
    ellipsoid = bound.bounds[0]
    rotation = np.linalg.qr(rng.normal(size=(27, 27)))[0]
    ellipsoid.B = (rotation * np.geomspace(.1, 1e-8, 27)) @ rotation.T
    ellipsoid.B_inv = np.linalg.inv(ellipsoid.B)
    ellipsoid.c = np.zeros(27)
    z = rng.normal(size=(4096, 27))
    z /= np.linalg.norm(z, axis=1)[:, None]
    z *= np.where(np.arange(len(z)) % 2, 1+1e-6, 1-1e-6)[:, None]
    points = ellipsoid.transform(z, inverse=True)
    stock = bound.contains(points)
    bound.backend = 'jax'
    np.testing.assert_array_equal(bound.contains(points), stock)
    assert np.sum(stock) == len(z)//2


def test_empty_candidate_guard(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.random((200, 2))
    y = -np.linalg.norm(x - .5, axis=1)
    bound = NautilusBound.compute(x, y, np.median(y), np.log(.5),
                                 n_networks=0, rng=rng, backend='numpy')
    monkeypatch.setattr(bound, '_contains_neural', lambda p: np.zeros(len(p), bool))
    bound.sample(1000, return_points=False, _max_draws=2000)
    assert bound.n_sample == bound.n_reject == 2000
    assert bound.log_v == np.inf


def test_nautilus_device_rng():
    pytest.importorskip('jax')
    rng = np.random.default_rng(0)
    x = rng.random((200, 2))
    y = -np.linalg.norm(x - .5, axis=1)
    bound = NautilusBound.compute(x, y, np.median(y), np.log(.5),
                                 n_networks=0, rng=rng, backend='jax')
    bound.rng = np.random.default_rng(123)
    state, outer_state = deepcopy(bound.rng.bit_generator.state), deepcopy(rng.bit_generator.state)
    points = bound.sample(17)
    assert bound.rng.bit_generator.state != state
    assert rng.bit_generator.state == outer_state
    assert np.all(bound.contains(points))
    assert bound.n_sample - bound.n_reject == 17 + len(bound.points)


def test_kway_fallback():
    rng = np.random.default_rng(12)
    x = np.vstack([rng.normal(center, .005, (100, 2)) for center in
                   [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]])
    bound = Union.compute(x, n_points_min=30, bound_class=UnitCubeEllipsoidMixture,
                          backend='numpy', rng=rng)
    volume = logsumexp(bound.log_v_all)
    bound.block[:] = True  # Binary splitting has exhausted this component.
    assert bound.split()
    assert len(bound.bounds) >= 3
    assert logsumexp(bound.log_v_all) < volume
    assert np.all(bound.contains(x))
    assert all(len(g) >= 30 for g in bound.points_bounds)


def test_exact_f_live_memo(monkeypatch):
    s = Sampler(lambda x:x, lambda x:0., n_dim=2, n_live=3, backend='numpy')
    s.log_l = [np.array([0., 0., 1., 2.]), np.array([1., 2., 2.])]
    s.shell_n = np.array([4, 3])
    s.shell_log_v = np.array([-1., -10.])

    def stock():
        log_l = np.concatenate(s.log_l)
        log_w = log_l + np.repeat(s.shell_log_v - np.log(s.shell_n), s.shell_n)
        return np.exp(logsumexp(log_w[np.argsort(log_l)][-s.n_live:]) - logsumexp(log_w))

    for change in ('initial', 'volumes', 'likelihoods', 'live'):
        if change == 'volumes':
            s.shell_log_v[0] -= 1
        elif change == 'likelihoods':
            s.log_l[1] = s.log_l[1] + 1
        elif change == 'live':
            s.n_live = 2
        assert s.f_live == stock()
    monkeypatch.setattr(np, 'argsort', lambda x: pytest.fail('memo missed'))
    assert s.f_live == s._f_live_value
