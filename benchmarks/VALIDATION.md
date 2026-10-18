CPU validation on 2026-10-17 against upstream `9d5fbc2`, with review fixes
after `006919a`. Used the fork's development Python, JAX 0.6.2, sklearn 1.7.2,
four CPU cores per process, and single-threaded BLAS/OMP. No GPU was available.
Production environments and source monkeypatches were not edited; nothing was
pushed. Git's existing automatic author identity was retained.

Full suites after the review fixes: **226 passed** on NumPy (1053.99 s) and
**226 passed** on JAX CPU (1084.71 s), with 27/39 warnings respectively
(polynomial conditioning and deliberately short sklearn fits). These include
197 existing tests and 29 added regression cases. `test_pool.py` remains
excluded as directed: upstream imports the absent optional `dask.distributed`
at collection. All other pool cases passed, using spawn to avoid forking after
JAX initializes. Syntax checks and `git diff --check` also passed.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NAUTILUS_TEST_BACKEND=numpy python -m pytest -q --ignore=tests/test_pool.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NAUTILUS_TEST_BACKEND=jax python -m pytest -q --ignore=tests/test_pool.py
```

The GPU-only auto default supersedes the original CPU auto default. Only
`Sampler` resolves auto; internal compute/read/train defaults are NumPy.
Explicit `backend="jax"` exercises JAX on CPU. Pools, fits below 4096 rows,
and unsupported sklearn training options retain stock code. Geometry uses
local float64 contexts, and each device proposal key comes from its bound's
saved NumPy RNG. CPU auto probing is quiet and restores the JAX logger state.
Nautilus sets the preallocation default before its first JAX entry point;
users initializing JAX earlier must set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

HDF5 writers are unchanged, including an AST comparison against `006919a`.
Tests cover checkpoint interchange in both directions. Earlier separate-process
checks with an unmodified `git archive 9d5fbc2 nautilus` tree resumed upstream
checkpoints with JAX and vice versa from 1500 to 3000 likelihood calls. Upstream
also read and predicted with a two-network JAX-trained ensemble. Pickles now
omit disposable f_live caches; the checkpoint format remains unchanged.

NumPy matches upstream bit-for-bit as long as neither the k-way split fallback
nor the empty-bound guard fires. Both safeguards remain enabled on both
backends; the split fallback consumes the bound's NumPy RNG. The
`bincount(minlength=2)` correction enforces the existing minimum cluster size
when a GMM leaves the highest-numbered cluster empty (previously unnoticed).
Ordinary two-cluster splits are unchanged. Original 500-live-point NumPy
posterior arrays and evidence were bit-identical to upstream in all nine
comparisons, under the same seeds/options.

The JAX trainer now permutes only valid indices, pads afterward, and performs
exactly `ceil(n / batch_size)` Adam steps per epoch. Epoch loss is the
sample-count-weighted mean of pre-update minibatch losses, as in sklearn.
Adam uses sklearn's bias-corrected rate with epsilon on `sqrt(v)`; equality
resets the stopping counter. Every fit starts with shuffle `PRNGKey(0)`;
sklearn initialization still uses network seeds 0, 1, ... without sampler RNG
consumption. New row counts can recompile training; inference and sampling
retain fixed chunks. Private initialization API `AttributeError`/`TypeError`
falls back to stock training.

A same-minibatch test matches sklearn's five steps, weights, and epoch loss for
37 rows / batch size 8. Three-network, 4097-row runs match initialization
bit-for-bit before conversion to float32 and execute 840 steps over 40 epochs.
Their ensemble-mean loss curves have mean absolute log-ratio 0.173 and
trailing-ten-epoch mean-loss ratio 1.389. Four injected private API
incompatibilities all select bit-identical stock training.

The original nine 500-live-point JAX toy runs were audited using the extracted
`006919a` package: **0 of 132 ensemble fits used the JAX trainer** (at most
1348 rows). They exercised JAX geometry, prediction, and sampling with
CPU-trained networks. The updated table below uses 18 native NumPy/JAX runs:
seeds 0, 1, 2, **2500 live points**, the real 4096-row training gate, two networks,
default 10000-epoch cap, `n_eff=3000`, `f_live=0.05`, and discarded exploration
samples. All nine JAX runs exercised accelerated training: 81 ensemble fits,
with two networks each. Gaussian/mixture normalizations include exact prior
truncation; the triangular 6-D banana reduces to 1-D quadrature.

| Toy | Reference lnZ | NumPy lnZ | JAX lnZ | JAX ensemble fits, seeds 0/1/2 |
|---|---:|---:|---:|---:|
| Gaussian, 12-D | -8.43596 | -8.42673 ± 0.03720 | -8.43119 ± 0.03075 | 11/11/11 |
| Bimodal mixture, 8-D | -9.66043 | -9.67197 ± 0.04761 | -9.65561 ± 0.04226 | 4/4/4 |
| Curved banana, 6-D | -6.62134 | -6.61403 ± 0.03550 | -6.62203 ± 0.02948 | 12/12/12 |

Entries are means over seeds; ± is the RMS per-run uncertainty diagnostic,
not the error on the mean. Nautilus has no native lnZ uncertainty output.
The script combines conditional shell importance-sampling variance with
binomial relative-volume variance from neural-bound counters. This is a
conservative approximate diagnostic, without covariance propagation or
calibrated coverage. Every individual lnZ error is below this diagnostic's one-sigma value (largest ratio 0.93).

Pooled weighted posterior comparison against native NumPy, worst coordinate:

| Toy | Mean difference / posterior SD | Relative SD difference | Max 16/50/84% quantile difference | Weighted KS D | Adjusted p |
|---|---:|---:|---:|---:|---:|
| Gaussian, 12-D | 0.0196 | 2.11% | 0.00766 | 0.02177 | 0.327 |
| Bimodal mixture, 8-D | 0.0260 | 1.56% | 0.01314 | 0.01616 | 1.000 |
| Curved banana, 6-D | 0.0301 | 0.98% | 0.00477 | 0.01620 | 1.000 |

Quantile differences use unit-prior coordinates. KS p-values use summed ESS
and a within-problem Bonferroni correction; they are approximate for weighted
adaptive samples. Raw JSON/NPZ results remain in `/tmp/nautilus-review-parity`;
the audit is in `/tmp/nautilus-review-old-toys`. Reproduce with
`python benchmarks/validate_backends.py --backend numpy --n-live 2500
--output /tmp/nautilus-review-parity` and then `--backend jax`. Actual JAX fit
counts are recorded only by the validation script, without production metadata.

The fused sampler compacts outer-union survivors on device, scores only those
survivors in padded 1024-point chunks, and returns once to the host. Raw-draw
counters, retained surplus, bound-local RNG, and volume accounting are
unchanged. A new device-independent regression compares contains on 1,000,000
points (half near ellipsoid surfaces), checks sampled points against both
predicates, and compares counter-based log_v within Monte Carlo error. The
existing 27-D surface test covers geometry with condition about 1e7.

`benchmarks/real_bound_parity.py` copies its input to `/tmp` before any HDF5
opening and opens only that copy, read-only. CPU checks used bounds 41–43
from a copy of `/scratch/midway3/tfairnington/stall15_20261017/ck15.hdf5`
(27-D, 5000 live points), with 4096 queries and 128 samples per bound. Masks
match exactly, every sample is contained under both predicates, and volume
differences are 1.22, 1.27, and 1.69 combined Monte Carlo sigma respectively.

| Frozen bound | NumPy log_v | JAX log_v | JAX CPU before (s) | Fused (s) | Speedup |
|---|---:|---:|---:|---:|---:|
| 41 | -31.4192 | -31.5678 | 2.3295 | 1.0487 | 2.22x |
| 42 | -32.0158 | -31.8613 | 1.6287 | 0.7082 | 2.30x |
| 43 | -32.1619 | -31.9562 | 1.0115 | 0.5206 | 1.94x |

Timings exclude compilation and compare the same script against an extracted
`006919a` package and this code. JAX raw draw counts (114688, 81920, 49152)
and log_v are identical before/after. Runs were sequential on the same four
cores with single-threaded BLAS; shared-node load limits timing precision. Reproduce with `python benchmarks/real_bound_parity.py
COPIED_CHECKPOINT --queries 4096 --samples 128`; on a GPU use
`--queries 1000000 --samples 10000`.

Refreshed CPU benchmark seconds, including compilation/transfers and profiler
overhead (the training gate still applies):

| Size/backend | Wall | Likelihood | Training | Bound sampling | Other |
|---|---:|---:|---:|---:|---:|
| 6-D NumPy | 13.825 | 0.554 | 5.213 | 0.272 | 7.786 |
| 6-D JAX CPU | 15.819 | 0.375 | 4.098 | 2.813 | 8.532 |
| 30-D NumPy | 23.523 | 1.190 | 4.907 | 0.365 | 17.061 |
| 30-D JAX CPU | 32.801 | 1.085 | 4.304 | 1.708 | 25.704 |

The small run used `--n-dim 6 --n-live 200 --n-networks 2 --n-like-max 2000`;
the larger used `--n-dim 30 --n-live 2500 --n-networks 2 --n-like-max 6000
--max-iter 80`, both with `--cost-ms .1`. Benchmark defaults remain 30-D,
5000 live points, 16 networks, and 10000 epochs for the GPU orchestrator.
These are shared-node smoke timings. CPU results support the NumPy default.

The f_live tests compare the incremental path against the exact memo path.
Six toy/seed combinations per backend (Gaussian, FP32, heavily tied plateaus;
seeds 0 and 5) have identical stopping iterations, posterior arrays, lnZ, and
RNG states. An adversarial tie case has fast estimate 0.25 versus exact around
7e-14; a conservative tied-weight bound also triggers exact confirmation.
The public f_live property is unchanged; status receives one estimate per batch.

`python benchmarks/benchmark_f_live.py` measured median milliseconds over three
changed batches (32 shells, 5000 live points, initially 8192 active-shell points):

| Synthetic points | Cold cache, fast | Changed batch, fast | Exact | Speedup |
|---|---:|---:|---:|---:|
| 1M | 30.250 | 2.932 | 65.824 | 22.5x |
| 4M | 73.348 | 2.869 | 332.749 | 116.0x |
| 8M | 127.022 | 2.775 | 851.317 | 306.7x |

Warm cost depends on changed-shell size plus `n_shells * n_live`, independently
of frozen-shell sizes. Exact sorting near convergence remains intentional;
resume first populates the cache. These timings precede the review fixes;
the arithmetic was unchanged in this review.

Against upstream, production/config totals +631/−62 (569 net lines); this review
changes existing production code by +99/−64 (35 net). The overall diff exceeds
800 net lines because it also includes backend/safeguard/f_live regressions,
reproducible benchmarks, and validation documentation. There are no production
monkeypatches, sidecar formats, duplicate stock methods, or dead reference code.

GPU throughput and full 35-D retrievals remain untested. Large late shells
whose networks cannot separate the top 0.5% of roughly a million points can
still stall; only the specified empty-candidate hang is guarded. Atomic full
writes and prefix-checked incremental shell writes remain deferred. Training
uses sklearn's private API (tested on 1.7.2), with the new fallback on drift.
