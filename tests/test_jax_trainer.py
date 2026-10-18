"""Check the supported trainer against the installed sklearn implementation."""
import subprocess
import sys

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from nautilus import neural


def test_valid_minibatches_adam_and_epoch_loss():
    jax = pytest.importorskip('jax')
    rng = np.random.default_rng(7)
    x = rng.normal(size=(37, 3)).astype('float32')
    y = np.sin(x[:, 0])
    reference = MLPRegressor(hidden_layer_sizes=(4,), alpha=0, shuffle=False,
                             learning_rate_init=.01, random_state=0)
    reference._random_state = np.random.RandomState(0)
    reference._initialize(y[:, None], [3, 4, 1], np.float32)
    parameters = tuple(tuple(w[None].copy() for w in layers) for layers in
                       (reference.coefs_, reference.intercepts_))
    key = jax.random.PRNGKey(0)
    weights, history, _, steps = neural._jax_trainer(1, 8, len(x))(
        parameters, key, np.pad(x, ((0, 3), (0, 0))), np.pad(y, (0, 3)),
        np.arange(40) < len(x), .01, 10, 0.)
    _, shuffle_key = jax.random.split(key)
    order = np.asarray(jax.random.permutation(shuffle_key, len(x)))
    accumulated = 0.
    for start in range(0, len(x), 8):
        indices = order[start:start+8]
        reference.partial_fit(x[indices], y[indices])
        accumulated += reference.loss_ * len(indices)
    assert int(steps) == reference._optimizer.t == 5
    np.testing.assert_allclose(history[0, 0], accumulated/len(x), rtol=1e-5)
    for actual, expected in zip(weights[0]+weights[1], reference.coefs_+reference.intercepts_):
        np.testing.assert_allclose(actual[0], expected, atol=1e-6, rtol=1e-5)


def test_initialization_steps_and_loss_trajectories(monkeypatch):
    jax = pytest.importorskip('jax')
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4097, 3))
    y = np.linalg.norm(x, axis=1)/5
    initialized, recorded = [], {}
    original_init, original_trainer = MLPRegressor._initialize, neural._jax_trainer

    def initialize(self, *args):
        original_init(self, *args)
        initialized.append([w.copy() for w in self.coefs_+self.intercepts_])

    def trainer(*args):
        run = original_trainer(*args)

        def fit(parameters, *values):
            recorded['initial'] = jax.device_get(parameters)
            result = run(parameters, *values)
            recorded['steps'] = int(result[3])
            return result

        return fit

    monkeypatch.setattr(MLPRegressor, '_initialize', initialize)
    monkeypatch.setattr(neural, '_jax_trainer', trainer)
    options = dict(n_networks=3, neural_network_kwargs={'max_iter':40, 'n_iter_no_change':100})
    accelerated = neural.NeuralNetworkEmulator.train(x, y, backend='jax', **options)
    stock = neural.NeuralNetworkEmulator.train(x, y, backend='numpy', **options)
    assert recorded['steps'] == 40*21
    for i, network in enumerate(stock.neural_networks):
        assert network._optimizer.t == recorded['steps']
        assert accelerated.neural_networks[i].t_ == network.t_ == 40*len(x)
        for actual, expected in zip(initialized[i], initialized[i+3]):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(recorded['initial'][0]+recorded['initial'][1], initialized[i]):
            np.testing.assert_array_equal(actual[i], expected.astype('float32'))
    a = np.mean([n.loss_curve_ for n in accelerated.neural_networks], axis=0)
    b = np.mean([n.loss_curve_ for n in stock.neural_networks], axis=0)
    assert np.mean(abs(np.log(a/b))) < .35
    assert .5 < np.mean(a[-10:])/np.mean(b[-10:]) < 2


@pytest.mark.parametrize('method', ['_initialize', '_validate_params'])
@pytest.mark.parametrize('error', [AttributeError, TypeError])
def test_private_api_fallback(method, error, monkeypatch):
    original = getattr(MLPRegressor, method)

    def incompatible(self, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == 'train':
            raise error('unsupported manual initialization')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MLPRegressor, method, incompatible)
    monkeypatch.setattr(neural, '_jax_trainer', lambda *args: pytest.fail('JAX should fall back'))
    x = np.random.default_rng(0).normal(size=(4096, 2))
    options = dict(n_networks=1, neural_network_kwargs={'max_iter':1})
    a = neural.NeuralNetworkEmulator.train(x, x[:, 0], backend='jax', **options)
    b = neural.NeuralNetworkEmulator.train(x, x[:, 0], backend='numpy', **options)
    assert hasattr(a.neural_networks[0], '_optimizer')
    for actual, expected in zip(a.neural_networks[0].coefs_, b.neural_networks[0].coefs_):
        np.testing.assert_array_equal(actual, expected)


def test_internal_defaults_do_not_probe_jax():
    code = '''
import inspect, sys, numpy as np
from nautilus import neural, bounds
from nautilus.bounds.basic import minimum_volume_enclosing_ellipsoid
functions = [minimum_volume_enclosing_ellipsoid]
for cls in (neural.NeuralNetworkEmulator, bounds.Ellipsoid, bounds.Union,
            bounds.NeuralBound, bounds.NautilusBound, bounds.UnitCubeEllipsoidMixture):
    functions += [getattr(cls, name) for name in ('train', 'compute', 'read')
                  if hasattr(cls, name) and 'backend' in inspect.signature(getattr(cls, name)).parameters]
assert all(inspect.signature(f).parameters['backend'].default == 'numpy' for f in functions)
minimum_volume_enclosing_ellipsoid(np.random.default_rng(0).normal(size=(30, 2)))
assert 'jax' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)
