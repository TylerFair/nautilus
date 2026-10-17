"""Run the existing suite with NAUTILUS_TEST_BACKEND=numpy or jax."""
import multiprocessing
import os

import pytest

from nautilus import neural, sampler
from nautilus.bounds import basic, nautilus, union
from nautilus.bounds import neural as neural_bound


if os.environ.get('NAUTILUS_TEST_BACKEND'):
    # Forking a process after JAX starts threads/CUDA is unsupported.
    multiprocessing.set_start_method('spawn', force=True)


@pytest.fixture(autouse=True)
def backend_for_tests(monkeypatch):
    selected = os.environ.get('NAUTILUS_TEST_BACKEND')
    if selected:
        original = neural.resolve_backend

        def resolve(backend, pool=None):
            return original(selected if backend == 'auto' else backend, pool)

        for module in (neural, sampler, basic, nautilus, union, neural_bound):
            monkeypatch.setattr(module, 'resolve_backend', resolve)
