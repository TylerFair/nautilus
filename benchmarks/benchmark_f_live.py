"""Changed-batch f_live timing with frozen shells and an 8192-point active shell."""
import argparse
import json
from time import perf_counter

import numpy as np

from nautilus import Sampler


def benchmark(n, repeats, n_live):
    rng = np.random.default_rng(0)
    sampler = Sampler(lambda x: x, lambda x: 0., n_dim=2,
                      n_live=n_live, backend='numpy')
    sizes = np.full(32, (n - 8192) // 31)
    sizes[-1] = 8192
    sizes[0] += n - np.sum(sizes)
    sampler.log_l = [rng.normal(size=int(size)) for size in sizes]
    sampler.shell_n = sizes
    sampler.shell_log_v = -np.linspace(0, 3, len(sizes))
    start = perf_counter()
    sampler._estimate_f_live()
    cold = perf_counter() - start
    fast_times, exact_times = [], []
    for _ in range(repeats):
        sampler.log_l[-1] = np.append(sampler.log_l[-1], rng.normal(size=64))
        sampler.shell_n[-1] += 64
        sampler.n_like += 64
        start = perf_counter()
        fast = sampler._estimate_f_live()
        fast_times.append(perf_counter() - start)
        start = perf_counter()
        exact = sampler.f_live
        exact_times.append(perf_counter() - start)
        np.testing.assert_allclose(fast, exact, rtol=1e-13)
    fast_ms, exact_ms = 1000*np.median(fast_times), 1000*np.median(exact_times)
    print(json.dumps(dict(points=n, cold_fast_ms=1000*cold, fast_ms=fast_ms,
                          exact_ms=exact_ms, speedup=exact_ms/fast_ms)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1_000_000, 4_000_000, 8_000_000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--n-live', type=int, default=5000)
    args = parser.parse_args()
    for size in args.sizes:
        benchmark(size, args.repeats, args.n_live)
