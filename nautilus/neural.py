"""Module implementing neural network emulators."""

import warnings
import os
from functools import lru_cache, partial

import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits


def resolve_backend(backend, pool=None):
    """Choose a backend, checking pools before importing/initializing JAX."""
    if backend not in ('auto', 'jax', 'numpy'):
        raise ValueError("backend must be 'auto', 'jax', or 'numpy'")
    if pool is not None:
        return 'numpy'
    if backend != 'numpy':
        try:
            os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            import jax
            return 'jax' if backend == 'jax' or any(
                d.platform == 'gpu' for d in jax.devices()) else 'numpy'
        except ImportError:
            if backend == 'jax':
                raise ImportError("backend='jax' requires JAX") from None
    return 'numpy'


def _jax_call(function, parameters, points):
    """Evaluate geometry in bounded float64 buckets without consuming RNG."""
    from jax.experimental import enable_x64
    points = np.asarray(points, dtype=float)
    shape = points.shape[:-1]
    points = np.atleast_2d(points)
    result = np.zeros(len(points), bool)
    with enable_x64():
        for start in range(0, len(points), 16384):
            chunk = points[start:start + 16384]
            size = next(n for n in (256, 1024, 4096, 16384) if n >= len(chunk))
            result[start:start + len(chunk)] = np.asarray(function(
                parameters, np.pad(chunk, ((0, size - len(chunk)), (0, 0)))))[:len(chunk)]
    return result.reshape(shape)


@lru_cache(maxsize=16)
def _jax_forward(activation='relu', output='identity'):
    import jax
    import jax.numpy as jnp

    activations = dict(identity=lambda x: x, relu=jax.nn.relu,
                       tanh=jnp.tanh, logistic=jax.nn.sigmoid, exp=jnp.exp)

    @jax.jit
    def forward(parameters, x):
        weights, biases, mean, scale = parameters
        values = (x - mean) / scale
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            values = values @ weight + bias[:, None, :]
            values = activations[activation if i < len(weights) - 1 else output](values)
        return jnp.mean(values, axis=0)[:, 0]

    return forward


@lru_cache(maxsize=16)
def _jax_trainer(max_iter, batch_size):
    """Cache executable shapes; data and stopping settings stay dynamic."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def train(params, key, x, y, valid, rate, patience, tol):
        n_networks = params[0][0].shape[0]
        def loss(params, indices, active):
            # Keep independent network predictions for independent gradients.
            weights, biases = params
            values = x[indices]
            for i, (w, b) in enumerate(zip(weights, biases)):
                values = values @ w + b[:, None, :]
                if i < len(weights) - 1:
                    values = jax.nn.relu(values)
            mask = valid[indices]
            losses = 0.5 * jnp.sum((values[:, :, 0] - y[indices])**2 * mask, axis=1)
            losses /= jnp.maximum(jnp.sum(mask, axis=1), 1)
            return jnp.sum(losses * active), losses

        zeros = jax.tree.map(jnp.zeros_like, params)
        initial = (params, zeros, zeros, 0, key,
                   jnp.full(n_networks, jnp.inf), jnp.zeros(n_networks, int),
                   jnp.ones(n_networks, bool), jnp.zeros(n_networks, int))

        def epoch(carry):
            epoch, state, history = carry
            params, moment, variance, step, key, best, stale, active, stopped = state
            key, *keys = jax.random.split(key, n_networks + 1)
            permutations = jax.vmap(lambda k: jax.random.permutation(k, len(x)))(
                jnp.stack(keys)).reshape(n_networks, -1, batch_size)

            def batch(state, indices):
                params, moment, variance, step = state
                (_, losses), gradients = jax.value_and_grad(loss, has_aux=True)(
                    params, indices, active)
                step += 1
                moment = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, moment, gradients)
                variance = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2,
                                       variance, gradients)
                proposed = jax.tree.map(
                    lambda p, m, v: p - rate * (m / (1 - 0.9**step)) /
                    (jnp.sqrt(v / (1 - 0.999**step)) + 1e-8), params, moment, variance)
                params = jax.tree.map(lambda p, q: jnp.where(
                    active.reshape((n_networks,) + (1,) * (p.ndim - 1)), q, p),
                    params, proposed)
                return (params, moment, variance, step), losses

            (params, moment, variance, step), losses = jax.lax.scan(
                batch, (params, moment, variance, step), permutations.transpose(1, 0, 2))
            losses = jnp.mean(losses, axis=0)
            stale = jnp.where(losses < best - tol, 0, stale + active)
            best = jnp.minimum(best, losses)
            stopped = jnp.where(active & (stale > patience), epoch + 1, stopped)
            active &= stale <= patience
            state = (params, moment, variance, step, key, best, stale, active, stopped)
            return epoch + 1, state, history.at[epoch].set(losses)

        _, state, history = jax.lax.while_loop(
            lambda c: (c[0] < max_iter) & jnp.any(c[1][7]), epoch,
            (0, initial, jnp.full((max_iter, n_networks), jnp.nan)))
        return state[0], history, jnp.where(state[7], max_iter, state[8])

    return train


@threadpool_limits.wrap(limits=1)
def train_network(x, y, neural_network_kwargs, random_state):
    """Train a network.

    Parameters
    ----------
    x : numpy.ndarray
        Input coordinates.
    y : numpy.ndarray
        Target values.
    neural_network_kwargs : dict
        Keyword arguments passed to the constructor of MLPRegressor.
    random_state : int
        Determines random number generation.

    Returns
    -------
    network : MLPRegressor
        The trained network.

    """
    return MLPRegressor(
        random_state=random_state, **neural_network_kwargs).fit(x, y)


class NeuralNetworkEmulator:
    """Likelihood neural network emulator.

    Attributes
    ----------
    mean : numpy.ndarray
        Mean of input coordinates used for normalizing coordinates.
    scale : numpy.ndarray
        Standard deviation of input coordinates used for normalizing
        coordinates.
    network : sklearn.neural_network.MLPRegressor
        Artifical neural network used for emulation.

    """

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None,
              backend='auto'):
        """Initialize and train the likelihood neural network emulator.

        Parameters
        ----------
        x : numpy.ndarray
            Input coordinates.
        y : numpy.ndarray
            Target values.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict or None, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor. Default is `None`.
        pool : multiprocessing.Pool, optional
            Pool used for parallel processing.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()
        emulator.backend = resolve_backend(backend, pool)

        emulator.mean = np.mean(x, axis=0)
        emulator.scale = np.std(x, axis=0)

        if neural_network_kwargs is None:
            neural_network_kwargs = {}

        neural_network_kwargs = dict(  # noqa: C408
            hidden_layer_sizes=(100, 50, 20), alpha=0, learning_rate_init=1e-2,
            max_iter=10000, tol=0, n_iter_no_change=10) | neural_network_kwargs

        if 'random_state' in neural_network_kwargs:
            warnings.warn("The 'random_state' keyword argument passed to the" +
                          " neural network is ignored.", Warning, stacklevel=2)
            neural_network_kwargs = neural_network_kwargs.copy()
            del neural_network_kwargs['random_state']

        # Unsupported sklearn configurations retain sklearn training/prediction.
        supported = dict(hidden_layer_sizes=(100, 50, 20), activation='relu',
                         solver='adam', alpha=0, learning_rate='constant',
                         shuffle=True, early_stopping=False, beta_1=0.9,
                         beta_2=0.999, epsilon=1e-8)
        options = set(supported) | {'learning_rate_init', 'max_iter', 'tol',
                                    'n_iter_no_change', 'batch_size'}
        use_jax = (emulator.backend == 'jax' and len(x) >= 4096 and
                   not (set(neural_network_kwargs) - options) and all(
                       neural_network_kwargs.get(k, v) == v for k, v in supported.items()))
        if use_jax:
            import jax
            import jax.numpy as jnp

            normalized = (x - emulator.mean) / emulator.scale
            networks = [MLPRegressor(random_state=i, **neural_network_kwargs)
                        for i in range(n_networks)]
            # sklearn's Glorot initialization keeps ensemble seeds compatible.
            for network in networks:
                if hasattr(network, '_validate_params'):
                    network._validate_params()
                network._random_state = np.random.RandomState(network.random_state)
                network._initialize(y[:, None], [x.shape[1], 100, 50, 20, 1], np.float64)
            params = tuple(tuple(jnp.asarray(np.stack([getattr(n, attr)[i]
                           for n in networks]), dtype=jnp.float32) for i in range(4))
                           for attr in ('coefs_', 'intercepts_'))
            batch_size = neural_network_kwargs.get('batch_size', 'auto')
            batch_size = min(200, len(x)) if batch_size == 'auto' else min(batch_size, len(x))
            capacity = max(256, 1 << (len(x) - 1).bit_length()) if len(x) < 8192 else (
                (len(x) + 8191) // 8192 * 8192)
            capacity = (capacity + batch_size - 1) // batch_size * batch_size
            x_pad = np.pad(normalized, ((0, capacity - len(x)), (0, 0)))
            y_pad = np.pad(y, (0, capacity - len(y)))
            weights, history, epochs = _jax_trainer(
                neural_network_kwargs['max_iter'], batch_size)(
                    params, jax.random.PRNGKey(0), jnp.asarray(x_pad, dtype=jnp.float32),
                    jnp.asarray(y_pad, dtype=jnp.float32), jnp.arange(capacity) < len(x),
                    neural_network_kwargs['learning_rate_init'],
                    neural_network_kwargs['n_iter_no_change'], neural_network_kwargs['tol'])
            weights, history, epochs = jax.device_get((weights, history, epochs))
            for i, network in enumerate(networks):
                network.coefs_ = [np.asarray(w[i], dtype=float) for w in weights[0]]
                network.intercepts_ = [np.asarray(b[i], dtype=float) for b in weights[1]]
                network.n_features_in_ = x.shape[1]
                network.n_iter_ = int(epochs[i])
                network.loss_curve_ = history[:epochs[i], i].astype(float).tolist()
                network.loss_ = network.loss_curve_[-1]
            emulator.neural_networks = networks
            return emulator

        f = partial(train_network, (x - emulator.mean) / emulator.scale, y,
                    neural_network_kwargs)

        if pool is None:
            emulator.neural_networks = list(map(f, range(n_networks)))
        else:
            emulator.neural_networks = list(pool.map(f, range(n_networks)))

        return emulator

    def predict(self, x):
        """Calculate the emulator likelihood prediction for a group of points.

        Parameters
        ----------
        x : numpy.ndarray
            Normalized coordinates of the training points.

        Returns
        -------
        y_emu : numpy.ndarray
            Emulated normalized likelihood value of the training points.

        """
        if getattr(self, 'backend', 'numpy') == 'jax' and len(x) >= 256:
            from jax.experimental import enable_x64
            with enable_x64():
                forward, params = self._jax_parameters()
                # Fixed chunks avoid a new compilation for each proposal count.
                result = [np.asarray(forward(params, np.pad(chunk,
                          ((0, 1024 - len(chunk)), (0, 0)))))[:len(chunk)]
                          for start in range(0, len(x), 1024)
                          for chunk in [x[start:start + 1024]]]
            return np.concatenate(result)
        return np.mean(
            [network.predict((x - self.mean) / self.scale) for network in
             self.neural_networks], axis=0)

    def _jax_parameters(self):
        """Cache device weights; the HDF5 writer only visits sklearn networks."""
        if not hasattr(self, '_parameters'):
            import jax.numpy as jnp
            networks = self.neural_networks
            self._parameters = (tuple(jnp.asarray(np.stack([n.coefs_[i] for n in networks]))
                                     for i in range(len(networks[0].coefs_))),
                                tuple(jnp.asarray(np.stack([n.intercepts_[i] for n in networks]))
                                     for i in range(len(networks[0].intercepts_))),
                                jnp.asarray(self.mean), jnp.asarray(self.scale))
        network = self.neural_networks[0]
        return _jax_forward(network.activation, network.out_activation_), self._parameters

    def write(self, group):
        """Write the emulator to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_networks'] = len(self.neural_networks)

        for i, network in enumerate(self.neural_networks):

            for key in network.__dict__:
                if key in ['coefs_', 'intercepts_']:
                    continue
                try:
                    group.attrs[key + f'_{i}'] = getattr(network, key)
                except (TypeError, ValueError):
                    pass

            for k in range(network.n_layers_ - 1):
                group.create_dataset(f'coefs_{k}_{i}', data=network.coefs_[k])
                group.create_dataset(
                    f'intercepts_{k}_{i}', data=network.intercepts_[k])

        group.create_dataset('mean', data=self.mean)
        group.create_dataset('scale', data=self.scale)

    @classmethod
    def read(cls, group, backend='auto'):
        """Read the emulator from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        Returns
        -------
        emulator : NeuralNetworkEmulator
            The likelihood neural network emulator.

        """
        emulator = cls()
        emulator.backend = resolve_backend(backend)

        emulator.mean = np.array(group['mean'])
        emulator.scale = np.array(group['scale'])

        emulator.neural_networks = []

        for i in range(group.attrs['n_networks']):

            network = MLPRegressor()

            for key in group.attrs:
                if key.rsplit('_', 1)[1] == f'{i}':
                    setattr(network, key.rsplit('_', 1)[0], group.attrs[key])

            network.coefs_ = [
                np.array(group[f'coefs_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]
            network.intercepts_ = [
                np.array(group[f'intercepts_{k}_{i}']) for k in
                range(network.n_layers_ - 1)]

            emulator.neural_networks.append(network)

        return emulator
