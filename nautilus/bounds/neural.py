"""Module implementing multi-dimensional neural network-based bounds."""

from functools import lru_cache

import numpy as np
from scipy.stats import rankdata

from ..neural import NeuralNetworkEmulator, _jax_call, _jax_forward
from .basic import Ellipsoid


@lru_cache(maxsize=16)
def _jax_neural(activations):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def contains(parameters, points):
        accepted = jnp.zeros(len(points), bool)
        for (c, inverse, emulator, threshold), activation in zip(parameters, activations):
            t = jnp.einsum('ij,nj->ni', inverse, points - c)
            inside = jnp.sum(t**2, axis=-1) < 1
            if emulator is not None:
                inside &= _jax_forward(*activation)(emulator, t) > threshold - 1e-9
            accepted |= inside
        return accepted

    return contains


def _jax_neural_parameters(bounds):
    import jax.numpy as jnp
    from jax.experimental import enable_x64
    parameters, activations = [], []
    with enable_x64():
        for bound in bounds:
            emulator = bound.emulator
            parameters.append((jnp.asarray(bound.outer_bound.c),
                               jnp.asarray(bound.outer_bound.B_inv),
                               None if emulator is None else emulator._jax_parameters()[1],
                               jnp.asarray(bound.score_predict_min)))
            network = None if emulator is None else emulator.neural_networks[0]
            activations.append(None if network is None else (
                network.activation, network.out_activation_))
    return _jax_neural(tuple(activations)), tuple(parameters)


class NeuralBound:
    """Neural network-based bound.

    Attributes
    ----------
    n_dim : int
        Number of dimensions.
    outer_bound : UnitCubeEllipsoidMixture
        Outer bound around the points above the likelihood threshold.
    emulator : object
        Emulator based on `sklearn.neural_network.MLPRegressor` used to fit and
        predict likelihood scores.
    score_predict_min : float
        Minimum score predicted by the emulator to be considered part of the
        bound.

    """

    @classmethod
    def compute(cls, points, log_l, log_l_min, enlarge_per_dim=1.1,
                n_networks=4, neural_network_kwargs=None, pool=None,
                rng=None, backend='numpy', warm_start=None):
        """Compute a neural network-based bound.

        Parameters
        ----------
        points : numpy.ndarray with shape (m, n)
            A 2-D array where each row represents a point.
        log_l : numpy.ndarray of length m
            Likelihood of each point.
        log_l_min : float
            Target likelihood threshold of the bound.
        enlarge_per_dim : float, optional
            Along each dimension, the ellipsoid of the outer bound is enlarged
            by this factor. Default is 1.1.
        n_networks : int, optional
            Number of networks used in the emulator. Default is 4.
        neural_network_kwargs : dict or None, optional
            Non-default keyword arguments passed to the constructor of
            MLPRegressor. Default is `None`.
        pool : nautilus.pool.NautilusPool or None, optional
            Pool used for parallel processing. Default is None.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.
        warm_start : NeuralNetworkEmulator or None, optional
            Emulator whose networks initialize JAX training. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()
        backend = backend if pool is None else 'numpy'
        bound.backend = backend
        bound.n_dim = points.shape[1]

        if rng is None:
            rng = np.random.default_rng()

        # Determine the outer bound.
        bound.outer_bound = Ellipsoid.compute(
            points[log_l >= log_l_min], enlarge_per_dim=enlarge_per_dim,
            rng=rng, backend=backend)

        if n_networks == 0:
            bound.emulator = None
            bound.score_predict_min = 0
            return bound

        # Train the network.
        if neural_network_kwargs is None:
            neural_network_kwargs = {}

        select = bound.outer_bound.contains(points)
        points = points[select]
        log_l = log_l[select]

        points_t = bound.outer_bound.transform(points)
        score = np.zeros(len(points))
        select = log_l >= log_l_min
        score[select] = 0.5 * (
            1 + (rankdata(log_l[select]) - 0.5) / np.sum(select))
        score[~select] = 0.5 * (
            (rankdata(log_l[~select]) - 0.5) / np.sum(~select))
        bound.emulator = NeuralNetworkEmulator.train(
            points_t, score, n_networks=n_networks,
            neural_network_kwargs=neural_network_kwargs, pool=pool, backend=backend,
            warm_start=warm_start)

        bound.score_predict_min = np.polyval(np.polyfit(
            score, bound.emulator.predict(points_t), 3),
            np.amin(score[select]))

        return bound

    def contains(self, points):
        """Check whether points are contained in the bound.

        Parameters
        ----------
        points : numpy.ndarray
            A 1-D or 2-D array containing single point or a collection of
            points. If more than one-dimensional, each row represents a point.

        Returns
        -------
        in_bound : bool or numpy.ndarray
            Bool or array of bools describing for each point if it is contained
            in the bound.

        """
        points = np.atleast_2d(points)
        if self.backend == 'jax':
            if not hasattr(self, '_parameters'):
                self._parameters = _jax_neural_parameters([self])
            return _jax_call(*self._parameters, points)

        in_bound = self.outer_bound.contains(points)
        if np.any(in_bound) and self.emulator is not None:
            points_t = self.outer_bound.transform(points)
            # In some instances, the network may predict practically the same
            # score for all input values. Lower the score threshold to prevent
            # no points being accepted in this case.
            in_bound[in_bound] = (self.emulator.predict(points_t[in_bound]) >
                                  self.score_predict_min - 1e-9)

        return in_bound

    def write(self, group):
        """Write the bound to an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.

        """
        group.attrs['n_dim'] = self.n_dim
        group.attrs['score_predict_min'] = self.score_predict_min
        self.outer_bound.write(group.create_group('outer_bound'))
        if self.emulator is not None:
            self.emulator.write(group.create_group('emulator'))

    @classmethod
    def read(cls, group, rng=None, backend='numpy'):
        """Read the bound from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to.
        rng : None or numpy.random.Generator, optional
            Determines random number generation. Default is None.

        Returns
        -------
        bound : NeuralBound
            The bound.

        """
        bound = cls()
        bound.backend = backend

        if rng is None:
            rng = np.random.default_rng()

        bound.n_dim = group.attrs['n_dim']
        bound.score_predict_min = group.attrs['score_predict_min']
        bound.outer_bound = Ellipsoid.read(group['outer_bound'], rng=rng)
        if 'emulator' in group:
            bound.emulator = NeuralNetworkEmulator.read(group['emulator'], backend=backend)
        else:
            bound.emulator = None

        return bound
