"""Compare cold and steady-state neural-bound costs on the active device."""

import argparse
import json
from time import perf_counter

import numpy as np

from nautilus.neural import NeuralNetworkEmulator
from nautilus.neural_jax import JaxNeuralNetworkEmulator, jax_device_info


def train(emulator_class, x, y, n_networks, kwargs):
    start = perf_counter()
    emulator = emulator_class.train(
        x, y, n_networks=n_networks, neural_network_kwargs=kwargs)
    elapsed = perf_counter() - start
    return emulator, elapsed


def predict(emulator, x, repeats):
    emulator.predict(x)
    start = perf_counter()
    for _ in range(repeats):
        prediction = emulator.predict(x)
    return prediction, (perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, default=20)
    parser.add_argument('--points', type=int, default=2000)
    parser.add_argument('--candidates', type=int, default=1000)
    parser.add_argument('--networks', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(4)
    x = rng.random((args.points, args.dimensions))
    radius = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(radius)) / float(len(radius))
    candidates = rng.random((args.candidates, args.dimensions))
    kwargs = {
        'max_iter': args.max_iter,
        'n_iter_no_change': 15,
    }

    numpy_emulator, numpy_train = train(
        NeuralNetworkEmulator, x, y, args.networks, kwargs)
    numpy_prediction, numpy_predict = predict(
        numpy_emulator, candidates, args.repeats)

    jax_emulator, jax_cold_train = train(
        JaxNeuralNetworkEmulator, x, y, args.networks, kwargs)
    # The second fit reuses compiled executables for the realistic steady-state
    # cost of subsequent Nautilus bounds in the same size bucket.
    jax_warm_emulator, jax_warm_train = train(
        JaxNeuralNetworkEmulator, x, y, args.networks, kwargs)
    jax_prediction, jax_predict = predict(
        jax_warm_emulator, candidates, args.repeats)

    output = {
        'device': jax_device_info(),
        'shape': {
            'dimensions': args.dimensions,
            'training_points': args.points,
            'candidate_points': args.candidates,
            'networks': args.networks,
        },
        'seconds': {
            'numpy_train': numpy_train,
            'numpy_predict_per_batch': numpy_predict,
            'jax_cold_train': jax_cold_train,
            'jax_warm_train': jax_warm_train,
            'jax_predict_per_batch': jax_predict,
        },
        'classification_agreement_at_0.5': float(np.mean(
            (numpy_prediction > 0.5) == (jax_prediction > 0.5))),
        'jax_prediction_finite': bool(np.all(np.isfinite(jax_prediction))),
        'jax_training_epochs': jax_emulator.n_iter_,
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
