"""Serial synthetic benchmark; includes compilation and transfers.

Example CPU smoke run: --n-dim 6 --n-live 200 --n-networks 2 --n-like-max 2000.
Set XLA_PYTHON_CLIENT_PREALLOCATE=false before launching on a shared GPU.
"""
import argparse
import cProfile
import json
import pstats
from time import perf_counter, sleep

import numpy as np

from nautilus import Sampler
from nautilus.neural import NeuralNetworkEmulator


def benchmark(args, backend):
    likelihood_seconds = 0.

    def likelihood(x):
        nonlocal likelihood_seconds
        start = perf_counter()
        if args.cost_ms:
            sleep(args.cost_ms / 1000)
        result = -np.sum((x - .5)**2) / (2 * .1**2)
        likelihood_seconds += perf_counter() - start
        return result

    start = perf_counter()
    sampler = Sampler(lambda x: x, likelihood, n_dim=args.n_dim,
                      n_live=args.n_live, n_networks=args.n_networks,
                      seed=args.seed, backend=backend,
                      neural_network_kwargs={'max_iter': args.max_iter})
    profile = cProfile.Profile()
    profile.enable()
    sampler.run(n_like_max=args.n_like_max, f_live=.2, n_eff=0)
    profile.disable()
    elapsed = perf_counter() - start
    stats = pstats.Stats(profile).stats
    training = sampling = 0.
    code = NeuralNetworkEmulator.train.__func__.__code__
    for (filename, line, name), (_, _, _, cumulative, callers) in stats.items():
        if filename == code.co_filename and line == code.co_firstlineno:
            training += cumulative
        if filename.endswith('/nautilus/sampler.py') and name == 'sample_shell':
            sampling += cumulative
        if filename.endswith('/bounds/nautilus.py') and name == 'sample':
            sampling += sum(v[3] for caller, v in callers.items() if caller[2] == 'add_bound')
    result = dict(backend=backend, n_like=sampler.n_like, bounds=len(sampler.bounds),
                  wall=elapsed, likelihood=likelihood_seconds, training=training,
                  bound_sampling=sampling,
                  other=elapsed - likelihood_seconds - training - sampling)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('n-dim', 30), ('n-live', 5000), ('n-networks', 16),
                          ('n-like-max', 30000), ('max-iter', 10000), ('seed', 0)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--cost-ms', type=float, default=.1)
    parser.add_argument('--backends', nargs='+', choices=['numpy', 'jax'],
                        default=['numpy', 'jax'])
    options = parser.parse_args()
    for selected in options.backends:
        benchmark(options, selected)
