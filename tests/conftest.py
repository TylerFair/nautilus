"""Run the existing suite with NAUTILUS_TEST_BACKEND=numpy or jax."""
import multiprocessing
import os

import pytest

from nautilus import neural, sampler
from nautilus.bounds import basic
from nautilus.bounds import NautilusBound, NeuralBound, Union


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

        monkeypatch.setattr(sampler, 'resolve_backend', resolve)
        functions = [basic.minimum_volume_enclosing_ellipsoid]
        for cls in (basic.Ellipsoid, basic.UnitCubeEllipsoidMixture, Union,
                    NeuralBound, NautilusBound, neural.NeuralNetworkEmulator):
            for name in ('compute', 'train', 'read'):
                function = getattr(cls, name, None)
                if function is not None and function.__func__.__defaults__[-1] == 'numpy':
                    functions.append(function.__func__)
        for function in functions:
            monkeypatch.setattr(function, '__defaults__', function.__defaults__[:-1] + (selected,))
