"""JAX implementation of the neural-network likelihood emulator.

JAX is imported lazily so that it remains an optional dependency of
Nautilus.  The implementation trains all ensemble members in one compiled
program and keeps their parameters on the selected JAX device for inference.
"""

import warnings

import numpy as np

_BATCH_SIZE = 256
_PREDICT_BUCKET_MIN = 128


def _import_jax():
    """Import JAX and return its public modules with an actionable error."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "The JAX neural-network backend requires JAX. Install Nautilus "
            "with the 'jax' extra and a JAX build appropriate for your "
            "accelerator.") from exc
    return jax, jnp


def jax_device_info():
    """Return a compact description of the active JAX device."""
    jax, _ = _import_jax()
    device = jax.devices()[0]
    return {
        'backend': jax.default_backend(),
        'platform': device.platform,
        'device': getattr(device, 'device_kind', str(device)),
    }


def _next_power_of_two(value):
    """Return the smallest power of two greater than or equal to value."""
    return 1 << max(0, int(value - 1).bit_length())


def _make_functions():
    """Construct compiled training and prediction functions lazily."""
    jax, jnp = _import_jax()

    def predict_one(params, x):
        value = x
        for i, (weight, bias) in enumerate(params):
            value = value @ weight + bias
            if i != len(params) - 1:
                value = jax.nn.relu(value)
        return value[..., 0]

    predict_ensemble = jax.jit(jax.vmap(
        predict_one, in_axes=(0, None), out_axes=0))

    def loss_one(params, x, y, mask, alpha):
        residual = predict_one(params, x) - y
        count = jnp.maximum(jnp.sum(mask), 1.0)
        mse = jnp.sum(mask * residual * residual) / count
        penalty = sum(jnp.sum(weight * weight) for weight, _ in params)
        return mse + alpha * penalty / count

    losses_and_grads = jax.vmap(
        jax.value_and_grad(loss_one),
        in_axes=(0, None, None, None, None), out_axes=(0, 0))

    @jax.jit
    def train_epoch(params, first_moment, second_moment, step, x, y, mask,
                    key, learning_rate, alpha, beta_1, beta_2, epsilon):
        order = jax.random.permutation(key, x.shape[0])
        x_epoch = x[order]
        y_epoch = y[order]
        mask_epoch = mask[order]
        n_batches = x.shape[0] // _BATCH_SIZE
        loss_init = jnp.zeros((next(iter(jax.tree.leaves(params))).shape[0],),
                              dtype=x.dtype)

        def train_batch(_, carry):
            (params_i, first_i, second_i, step_i, loss_sum_i,
             count_sum_i, offset_i) = carry
            x_batch = jax.lax.dynamic_slice_in_dim(
                x_epoch, offset_i, _BATCH_SIZE)
            y_batch = jax.lax.dynamic_slice_in_dim(
                y_epoch, offset_i, _BATCH_SIZE)
            mask_batch = jax.lax.dynamic_slice_in_dim(
                mask_epoch, offset_i, _BATCH_SIZE)
            count = jnp.sum(mask_batch)

            def update(values):
                params_u, first_u, second_u, step_u = values
                losses, grads = losses_and_grads(
                    params_u, x_batch, y_batch, mask_batch, alpha)
                step_u = step_u + 1
                first_u = jax.tree.map(
                    lambda m, g: beta_1 * m + (1.0 - beta_1) * g,
                    first_u, grads)
                second_u = jax.tree.map(
                    lambda v, g: beta_2 * v + (1.0 - beta_2) * g * g,
                    second_u, grads)
                correction_1 = 1.0 - beta_1**step_u
                correction_2 = 1.0 - beta_2**step_u
                params_u = jax.tree.map(
                    lambda p, m, v: p - learning_rate *
                    (m / correction_1) /
                    (jnp.sqrt(v / correction_2) + epsilon),
                    params_u, first_u, second_u)
                return params_u, first_u, second_u, step_u, losses

            def skip(values):
                params_s, first_s, second_s, step_s = values
                return params_s, first_s, second_s, step_s, loss_init

            params_i, first_i, second_i, step_i, losses = jax.lax.cond(
                count > 0,
                update,
                skip,
                (params_i, first_i, second_i, step_i))
            return (params_i, first_i, second_i, step_i,
                    loss_sum_i + losses * count, count_sum_i + count,
                    offset_i + _BATCH_SIZE)

        carry = (params, first_moment, second_moment, step, loss_init,
                 jnp.asarray(0.0, dtype=x.dtype),
                 jnp.asarray(0, dtype=jnp.int32))
        carry = jax.lax.fori_loop(0, n_batches, train_batch, carry)
        (params, first_moment, second_moment, step, loss_sum,
         count_sum, _) = carry
        return (params, first_moment, second_moment, step,
                loss_sum / jnp.maximum(count_sum, 1.0))

    return predict_ensemble, train_epoch


_FUNCTIONS = None


def _functions():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        _FUNCTIONS = _make_functions()
    return _FUNCTIONS


class JaxNeuralNetworkEmulator:
    """Likelihood neural-network ensemble trained and evaluated with JAX."""

    backend = 'jax'

    @property
    def n_networks(self):
        """Number of members in the ensemble."""
        return int(self.params[0][0].shape[0])

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs=None, pool=None):
        """Train every network in the ensemble in one compiled JAX program."""
        if pool is not None:
            warnings.warn(
                "The JAX backend trains the ensemble on one accelerator; "
                "the sampler pool is not used for neural-network fitting.",
                RuntimeWarning, stacklevel=2)

        jax, jnp = _import_jax()
        emulator = cls()
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("Training inputs and targets must be non-empty "
                             "and have matching lengths.")
        if n_networks < 1:
            raise ValueError("'n_networks' must be at least one.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("JAX neural-network training data must be finite.")
        emulator.mean = np.mean(x, axis=0)
        emulator.scale = np.std(x, axis=0)
        emulator.scale = np.where(emulator.scale == 0, 1.0, emulator.scale)

        defaults = {
            'hidden_layer_sizes': (100, 50, 20),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0,
            'batch_size': 'auto',
            'learning_rate_init': 1e-2,
            'max_iter': 10000,
            'tol': 0.0,
            'n_iter_no_change': 10,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'shuffle': True,
            'random_state': None,
            'verbose': False,
        }
        supplied = {} if neural_network_kwargs is None else dict(
            neural_network_kwargs)
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            raise ValueError(
                "Unsupported neural_network_kwargs for the JAX backend: " +
                ', '.join(unknown))
        config = defaults | supplied
        if config['activation'] != 'relu' or config['solver'] != 'adam':
            raise ValueError(
                "The JAX backend currently supports activation='relu' and "
                "solver='adam' only.")
        if config['batch_size'] not in ('auto', _BATCH_SIZE):
            raise ValueError(
                f"The JAX backend uses batch_size={_BATCH_SIZE}; pass 'auto' "
                f"or {_BATCH_SIZE}.")
        if not config['shuffle']:
            raise ValueError(
                "The JAX backend currently requires shuffle=True.")
        if config['random_state'] is not None:
            warnings.warn(
                "The 'random_state' neural-network keyword is ignored. "
                "Ensemble members use deterministic, distinct seeds.",
                Warning, stacklevel=2)

        hidden = config['hidden_layer_sizes']
        if isinstance(hidden, int):
            hidden = (hidden,)
        layer_sizes = (x.shape[1],) + tuple(hidden) + (1,)
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, n_networks * (len(layer_sizes) - 1))
        params = []
        for layer, (fan_in, fan_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_keys = keys[layer * n_networks:(layer + 1) * n_networks]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = jnp.stack([
                jax.random.uniform(
                    layer_key, (fan_in, fan_out), minval=-limit,
                    maxval=limit, dtype=jnp.float32)
                for layer_key in layer_keys
            ])
            biases = jnp.zeros((n_networks, fan_out), dtype=jnp.float32)
            params.append((weights, biases))
        params = tuple(params)

        x_normalized = ((x - emulator.mean) / emulator.scale).astype(
            np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        bucket_size = max(_BATCH_SIZE, _next_power_of_two(len(x)))
        bucket_size = int(np.ceil(bucket_size / _BATCH_SIZE) * _BATCH_SIZE)
        x_padded = np.zeros((bucket_size, x.shape[1]), dtype=np.float32)
        y_padded = np.zeros(bucket_size, dtype=np.float32)
        mask = np.zeros(bucket_size, dtype=np.float32)
        x_padded[:len(x)] = x_normalized
        y_padded[:len(y)] = y
        mask[:len(x)] = 1.0
        x_device = jax.device_put(x_padded)
        y_device = jax.device_put(y_padded)
        mask_device = jax.device_put(mask)

        first_moment = jax.tree.map(jnp.zeros_like, params)
        second_moment = jax.tree.map(jnp.zeros_like, params)
        step = jnp.asarray(0, dtype=jnp.int32)
        _, train_epoch = _functions()
        best_loss = np.inf
        stale_epochs = 0
        min_delta = max(float(config['tol']), 1e-7)
        max_iter = int(config['max_iter'])
        if max_iter < 1 or int(config['n_iter_no_change']) < 1:
            raise ValueError(
                "'max_iter' and 'n_iter_no_change' must be positive.")
        for epoch in range(max_iter):
            epoch_key = jax.random.fold_in(key, epoch + 1)
            (params, first_moment, second_moment, step,
             losses) = train_epoch(
                params, first_moment, second_moment, step,
                x_device, y_device, mask_device, epoch_key,
                float(config['learning_rate_init']), float(config['alpha']),
                float(config['beta_1']), float(config['beta_2']),
                float(config['epsilon']))
            loss = float(np.mean(np.asarray(losses)))
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "JAX neural-network training diverged to a non-finite "
                    "loss. Reduce learning_rate_init or use the NumPy "
                    "backend.")
            if best_loss - loss > min_delta:
                best_loss = loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            if config['verbose']:
                print(f'JAX neural epoch {epoch + 1}: loss={loss:.6g}')
            if stale_epochs >= int(config['n_iter_no_change']):
                break
        else:
            warnings.warn(
                "JAX neural-network training reached max_iter without "
                "converging.", RuntimeWarning, stacklevel=2)

        if not all(np.all(np.isfinite(np.asarray(value)))
                   for value in jax.tree.leaves(params)):
            raise FloatingPointError(
                "JAX neural-network training produced non-finite parameters. "
                "Reduce learning_rate_init or use the NumPy backend.")

        emulator.params = params
        emulator.n_iter_ = epoch + 1
        emulator.loss_ = best_loss
        emulator.config = config
        emulator.device_info = jax_device_info()
        return emulator

    def predict(self, x):
        """Predict the ensemble-mean score for a collection of points."""
        jax, _ = _import_jax()
        x = np.atleast_2d(np.asarray(x))
        x = ((x - self.mean) / self.scale).astype(np.float32, copy=False)
        bucket_size = max(_PREDICT_BUCKET_MIN, _next_power_of_two(len(x)))
        x_padded = np.zeros((bucket_size, x.shape[1]), dtype=np.float32)
        x_padded[:len(x)] = x
        predict_ensemble, _ = _functions()
        predictions = predict_ensemble(self.params, jax.device_put(x_padded))
        return np.asarray(predictions[:, :len(x)]).mean(axis=0)

    def write(self, group):
        """Write parameters in a backend-neutral HDF5 representation."""
        group.attrs['compute_backend'] = self.backend
        group.attrs['n_networks'] = self.n_networks
        group.attrs['n_layers'] = len(self.params)
        group.attrs['n_iter'] = self.n_iter_
        group.attrs['loss'] = self.loss_
        group.create_dataset('mean', data=self.mean)
        group.create_dataset('scale', data=self.scale)
        for layer, (weight, bias) in enumerate(self.params):
            group.create_dataset(f'weights_{layer}', data=np.asarray(weight))
            group.create_dataset(f'biases_{layer}', data=np.asarray(bias))

    @classmethod
    def read(cls, group):
        """Restore a JAX ensemble from an HDF5 group."""
        jax, _ = _import_jax()
        emulator = cls()
        emulator.mean = np.array(group['mean'])
        emulator.scale = np.array(group['scale'])
        emulator.params = tuple(
            (jax.device_put(np.array(group[f'weights_{layer}'])),
             jax.device_put(np.array(group[f'biases_{layer}'])))
            for layer in range(int(group.attrs['n_layers'])))
        emulator.n_iter_ = int(group.attrs.get('n_iter', 0))
        emulator.loss_ = float(group.attrs.get('loss', np.nan))
        emulator.config = {}
        emulator.device_info = jax_device_info()
        return emulator
