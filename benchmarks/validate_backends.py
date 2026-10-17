"""Reproducible toy parity runs; use PYTHONPATH for an unmodified stock tree."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, logsumexp, ndtr

from nautilus import Sampler
from nautilus.neural import NeuralNetworkEmulator


def problem(name):
    if name == 'gaussian':
        d, sigma = 12, .2
        likelihood = lambda x: -.5 * np.sum(((x - .5) / sigma)**2)
        truth = d * np.log(sigma * np.sqrt(2*np.pi) * erf(.5 / (np.sqrt(2)*sigma)))
    elif name == 'bimodal':
        d, sigma = 8, .12
        centers = np.array([np.full(d, .3), np.full(d, .7)])
        likelihood = lambda x: logsumexp(-.5*np.sum(((x-centers)/sigma)**2, axis=1))-np.log(2)
        truth = d * np.log(sigma*np.sqrt(2*np.pi) * (ndtr(.7/sigma)-ndtr(-.3/sigma)))
    else:
        d, sigma = 6, .14

        def likelihood(x):
            z = (x - .5) / sigma
            z[1] = (x[1] - .5 - 1.5*((x[0]-.5)**2-sigma**2)) / .1
            return -.5*np.sum(z*z)

        def marginal(x):
            center = .5 + 1.5*((x-.5)**2-sigma**2)
            return np.exp(-.5*((x-.5)/sigma)**2) * .1*np.sqrt(2*np.pi) * (
                ndtr((1-center)/.1) - ndtr(-center/.1))

        truth = np.log(quad(marginal, 0, 1, epsabs=1e-12)[0]) + (d-2)*np.log(
            sigma*np.sqrt(2*np.pi)*erf(.5/(np.sqrt(2)*sigma)))
    return d, likelihood, truth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['stock', 'numpy', 'jax'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--n-live', type=int, default=500)
    parser.add_argument('--n-eff', type=int, default=3000)
    parser.add_argument('--cases', nargs='+', choices=['gaussian', 'bimodal', 'curved'],
                        default=['gaussian', 'bimodal', 'curved'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        for seed in args.seeds:
            d, likelihood, truth = problem(name)
            start = perf_counter()
            options = {} if args.backend == 'stock' else {'backend': args.backend}
            sampler = Sampler(lambda x:x, likelihood, n_dim=d, n_live=args.n_live,
                              n_networks=2, seed=seed, **options)
            fits = []

            def record_fit(frame, event, result):
                if event == 'return' and frame.f_code is NeuralNetworkEmulator.train.__func__.__code__:
                    fits.append((len(frame.f_locals['x']), bool(frame.f_locals.get('use_jax'))))

            sys.setprofile(record_fit)
            try:
                sampler.run(f_live=.05, n_eff=args.n_eff, discard_exploration=True)
            finally:
                sys.setprofile(None)
            x, log_w, _ = sampler.posterior()
            w = np.exp(log_w)
            mean = w @ x
            std = np.sqrt(w @ ((x-mean)**2))
            quantiles = [np.interp([.16, .5, .84], np.cumsum(w[idx]), x[idx, j]).tolist()
                         for j in range(d) for idx in [np.argsort(x[:, j])]]
            # Conditional shell MC error, plus bound-volume binomial error.
            mass = np.exp(sampler.shell_log_l + sampler.shell_log_v - sampler.log_z)
            valid = sampler.shell_n_eff > 0
            variance = np.sum(mass[valid]**2 * np.maximum(0,
                1/sampler.shell_n_eff[valid] - 1/sampler.shell_n[valid]))
            volume_variance = sum(b.n_reject / max(1, b.n_sample * (b.n_sample-b.n_reject))
                                  for b in sampler.bounds[1:])
            result = dict(case=name, backend=args.backend, seed=seed, truth=truth,
                          log_z=sampler.log_z, conditional_sigma=float(np.sqrt(variance)),
                          volume_sigma=float(np.sqrt(volume_variance)), mean=mean.tolist(),
                          std=std.tolist(), quantiles=quantiles, n_like=sampler.n_like,
                          n_eff=sampler.n_eff, seconds=perf_counter()-start)
            result.update(jax_fits=sum(j for n, j in fits), fit_rows=[n for n, j in fits])
            stem = args.output / f'{name}-{args.backend}-{seed}'
            np.savez_compressed(str(stem)+'.npz', x=x, w=w)
            Path(str(stem)+'.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
