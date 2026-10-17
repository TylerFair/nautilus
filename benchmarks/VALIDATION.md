CPU validation on 2026-10-17 against upstream `9d5fbc2`.

Used `/scratch/midway3/tfairnington/nautilus_fork/devenv/bin/python`, JAX 0.6.2,
scikit-learn 1.7.2, four CPU cores per process, and single-threaded BLAS/OMP.
No GPU was available. Production environments and the source monkeypatches
were not edited; nothing was pushed. Git had no explicit name/email, so its
existing automatic identity was retained without an override.

The addendum's GPU-only auto default supersedes the original CPU auto default.
Explicit `backend="jax"` exercised JAX on CPU. Pools, fits below 4096 rows,
and unsupported sklearn training options retain stock code. Geometry uses
local float64 contexts; device proposal keys come from each bound's NumPy RNG.

Full suite: NumPy **206 passed in 801.78 s**; JAX CPU **206 passed in 807.17 s**.
JAX emitted ten `RankWarning: Polyfit may be poorly conditioned` warnings.
The two full runs collected 197 existing tests and nine new regressions.
Two subsequent regression cases cover a conditioned 27-D surface and separate
Nautilus/outer-union generators. Final focused runs passed all 11 regressions
on NumPy (36.69 s) and JAX (35.37 s), covering 208 unique cases per selection
across the full and focused runs.
`test_pool.py` was excluded as directed: the unchanged upstream file imports
the absent optional `dask.distributed` at collection. Other pool tests run,
using spawn in the test harness to avoid forking after JAX initializes.

Run the suite sequentially from the repository with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NAUTILUS_TEST_BACKEND=numpy python -m pytest -q --ignore=tests/test_pool.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NAUTILUS_TEST_BACKEND=jax python -m pytest -q --ignore=tests/test_pool.py
```

Checkpoint checks include both directions between the native backends and
separate processes using an unmodified `git archive 9d5fbc2 nautilus` tree.
Upstream resumed a JAX checkpoint from 1500 to 3000 likelihood calls, and JAX
resumed an upstream checkpoint over the same interval. Upstream also read and
predicted with a two-network JAX-trained ensemble. HDF5 writers are unchanged.
The exact `f_live` memo is tested against stock arithmetic, including ties.

Toy runs used seeds 0, 1, 2, `n_live=500`, two networks, `n_eff=3000`,
`f_live=0.05`, and discarded exploration samples. There were 27 runs: upstream,
native NumPy, and JAX for each seed/problem. Native NumPy evidence and weighted
posterior arrays were bit-identical to upstream in all nine comparisons.
Gaussian and mixture normalizations include exact prior truncation. The 6-D
banana uses a triangular bend; its normalization reduces to 1-D quadrature.

| Toy | Reference lnZ | Upstream/native NumPy lnZ | JAX lnZ |
|---|---:|---:|---:|
| Gaussian, 12-D | -8.43596 | -8.43957 ± 0.05880 | -8.43716 ± 0.03545 |
| Bimodal mixture, 8-D | -9.66043 | -9.66465 ± 0.07891 | -9.66227 ± 0.06585 |
| Curved banana, 6-D | -6.62134 | -6.60650 ± 0.05952 | -6.61137 ± 0.03004 |

Entries are means over seeds; ± is the RMS per-run uncertainty diagnostic,
not the error on the mean. Nautilus has no native lnZ uncertainty output.
The script combines conditional shell importance-sampling variance with
binomial relative-volume variance from every neural bound's counters. This
is a conservative approximate diagnostic, without covariance propagation or
calibrated coverage. Every individual lnZ error is below this diagnostic's
one-sigma value (largest ratio 0.83). Across-seed lnZ standard deviations are
0.00594/0.02518, 0.00810/0.00655, and 0.00324/0.00459 for NumPy/JAX respectively.
The curved toy's positive evidence offset also occurs upstream; its small
three-seed scatter alone would understate the reference discrepancy.

Pooled weighted posterior comparison against upstream, worst coordinate:

| Toy | Mean difference / posterior SD | Relative SD difference | Max 16/50/84% quantile difference | Weighted KS D | Adjusted p |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0.0461 | 1.70% | 0.01388 | 0.02548 | 0.063 |
| Bimodal | 0.0220 | 1.38% | 0.00894 | 0.01824 | 0.773 |
| Curved | 0.0305 | 1.01% | 0.00565 | 0.01663 | 0.942 |

Quantile differences are in unit-prior coordinates. KS p-values use summed
effective sample sizes and a within-problem Bonferroni correction; they are
approximate for weighted adaptive samples. All native NumPy differences are
zero. Raw per-run JSON/NPZ files remain in `/tmp/nautilus-parity`.

Reproduce with `python benchmarks/validate_backends.py --backend numpy
--output /tmp/nautilus-parity` and then `--backend jax`. For upstream use the
extracted tree on `PYTHONPATH` with `--backend stock`.

CPU benchmark seconds, including compilation/transfers and profiler overhead:

| Size/backend | Wall | Likelihood | Training | Bound sampling | Other |
|---|---:|---:|---:|---:|---:|
| 6-D NumPy | 12.404 | 0.469 | 4.362 | 0.323 | 7.249 |
| 6-D JAX CPU | 13.018 | 0.371 | 3.888 | 0.907 | 7.852 |
| 30-D NumPy | 46.593 | 2.655 | 11.385 | 0.583 | 31.969 |
| 30-D JAX CPU | 86.181 | 2.440 | 13.810 | 1.174 | 68.757 |

The small run used `--n-dim 6 --n-live 200 --n-networks 2 --n-like-max 2000`.
The larger run used `--n-dim 30 --n-live 2500 --n-networks 2 --n-like-max 6000
--max-iter 80`. Both used `--cost-ms .1`. Other validation jobs ran concurrently
on the shared login node. These are smoke timings, not isolated speed claims.
The benchmark defaults remain 30-D, 5000 live points, 16 networks, and 10000
epochs for the GPU orchestrator. CPU results support retaining the stock default.

The initial port added 480 net lines in existing modules/config.
Its tests and benchmark scripts added 373 lines; documentation accounts for the
remaining diff beyond the approximate 800-line budget. There are no production
monkeypatches, sidecar formats, or duplicate stock methods.

GPU throughput and full 35-D retrievals remain untested here. The 27-D surface
regression does exercise geometry with condition about 1e7. Large late shells
whose networks cannot separate the top 0.5% of roughly one million points can
still stall progress; only the specified empty-candidate hang is guarded.
Atomic full writes and prefix-checked incremental shell writes are deferred.
The trainer uses sklearn's private initialization API, tested here on 1.7.2.
Ruff was unavailable; an isolated `/tmp` install failed because network DNS
was unavailable. Syntax checks and `git diff --check` passed.

The f_live addendum compares the incremental path against the exact-memo path
from `f141cb5`. Six toy/seed combinations per backend (Gaussian, FP32, and
heavily tied plateaus; seeds 0 and 5) have identical stopping iterations,
posterior arrays, lnZ, and RNG states. A separate adversarial tie case has a
fast estimate of 0.25 versus an exact value around 7e-14: confirmation also uses
a conservative lower bound on tied weights, so a factor-two miss cannot delay
the exact stop. The public `f_live` property and checkpoint writers are unchanged.
The run loop computes one estimate per batch and passes it to status output.

Full suite reruns: **217 passed** on NumPy (488.75 s) and JAX CPU (489.51 s).
There were 16/28 polynomial-conditioning warnings respectively. The optional
Dask pool module remains excluded as directed; all other pool cases passed.

`python benchmarks/benchmark_f_live.py` reports median milliseconds over three
changed batches, with 32 shells, `n_live=5000`, and an initially 8192-point active
shell. Fast timings measure the incremental path outside exact confirmation.

| Synthetic points | Cold cache, fast | Changed batch, fast | Exact | Speedup |
|---|---:|---:|---:|---:|
| 1M | 30.250 | 2.932 | 65.824 | 22.5x |
| 4M | 73.348 | 2.869 | 332.749 | 116.0x |
| 8M | 127.022 | 2.775 | 851.317 | 306.7x |

Warm cost scales with changed-shell size plus `n_shells * n_live`, independently
of frozen-shell point counts. Near the stopping threshold, exact sorting remains
intentional; the first call after resume must populate the shell cache.
