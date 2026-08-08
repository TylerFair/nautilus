import numpy as np
import pytest

from nautilus import neural


def test_neural_network_emulator():
    # Test that the emulator works and produces decent estimates.

    np.random.seed(0)
    n_dim, n_points = 5, 1000
    x = np.random.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = neural.NeuralNetworkEmulator.train(x, y, n_networks=1, pool=None)
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)


def test_resolve_compute_backend():
    assert neural.resolve_compute_backend('numpy') == 'numpy'
    with pytest.raises(ValueError):
        neural.resolve_compute_backend('invalid')


@pytest.mark.parametrize('n_networks', [1, 4])
def test_jax_neural_network_emulator(n_networks):
    pytest.importorskip('jax')
    from nautilus.neural_jax import JaxNeuralNetworkEmulator

    rng = np.random.default_rng(0)
    n_dim, n_points = 5, 1000
    x = rng.random((n_points, n_dim))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = JaxNeuralNetworkEmulator.train(
        x, y, n_networks=n_networks,
        neural_network_kwargs={'max_iter': 300, 'n_iter_no_change': 15})
    prediction = emu.predict(x)

    assert prediction.shape == y.shape
    assert np.all(np.isfinite(prediction))
    assert np.sqrt(np.mean((y - prediction)**2)) < 0.3 * np.std(y)
    assert emu.n_networks == n_networks


def test_jax_and_numpy_neural_emulators_have_classification_parity():
    pytest.importorskip('jax')
    from nautilus.neural_jax import JaxNeuralNetworkEmulator

    rng = np.random.default_rng(2)
    x = rng.random((1200, 6))
    radius = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(radius)) / float(len(radius))
    kwargs = {'max_iter': 300, 'n_iter_no_change': 15}
    numpy_emu = neural.NeuralNetworkEmulator.train(
        x, y, n_networks=2, neural_network_kwargs=kwargs)
    jax_emu = JaxNeuralNetworkEmulator.train(
        x, y, n_networks=2, neural_network_kwargs=kwargs)

    threshold = 0.5
    numpy_class = numpy_emu.predict(x) > threshold
    jax_class = jax_emu.predict(x) > threshold
    assert np.mean(numpy_class == jax_class) > 0.95


def test_jax_neural_emulator_checkpoint(tmp_path):
    pytest.importorskip('jax')
    h5py = pytest.importorskip('h5py')
    from nautilus.neural_jax import JaxNeuralNetworkEmulator

    rng = np.random.default_rng(3)
    x = rng.random((500, 3))
    y = np.sum((x - 0.5)**2, axis=1)
    emulator = JaxNeuralNetworkEmulator.train(
        x, y, n_networks=2,
        neural_network_kwargs={
            'max_iter': 200, 'n_iter_no_change': 10, 'tol': 1e-5,
        })
    expected = emulator.predict(x[:37])
    filepath = tmp_path / 'emulator.h5'
    with h5py.File(filepath, 'w') as stream:
        emulator.write(stream.create_group('emulator'))
    with h5py.File(filepath, 'r') as stream:
        restored = neural.read_neural_network_emulator(stream['emulator'])

    assert restored.backend == 'jax'
    assert np.allclose(restored.predict(x[:37]), expected)
