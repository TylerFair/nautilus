"""Checkpoint failure safety and exact incremental-resume regression tests."""

import hashlib

import h5py
import numpy as np
import pytest

from nautilus import Sampler
from nautilus.sampler import _append_rows


def make_sampler(path=None, resume=False):
    """Use nonconstant weights and structured blobs across several batches."""
    return Sampler(
        lambda x: x,
        lambda x: (-np.sum((x - 0.5)**2), x[0], int(x[1] > 0.5)),
        n_dim=3, n_live=30, n_networks=0, n_batch=16, seed=13,
        backend='numpy', filepath=path, resume=resume)


def assert_hdf5_equal(left, right):
    """Compare values, ignoring physical chunk layout and file allocation."""
    def compare(name, obj):
        other = right[name]
        assert set(obj.attrs) == set(other.attrs)
        for key in obj.attrs:
            np.testing.assert_array_equal(obj.attrs[key], other.attrs[key])
        if isinstance(obj, h5py.Dataset):
            np.testing.assert_array_equal(obj[...], other[...])
    left.visititems(compare)


def test_atomic_write_preserves_checkpoint_after_error(tmp_path, monkeypatch):
    path = tmp_path / 'run.h5'
    sampler = make_sampler(path)
    sampler.run(n_like_max=64)
    original = hashlib.sha256(path.read_bytes()).digest()

    def fail(stream):
        stream.create_group('partial')
        raise OSError('simulated full disk')

    monkeypatch.setattr(sampler, '_write', fail)
    with pytest.raises(OSError, match='simulated full disk'):
        sampler.write(path, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).digest() == original
    assert list(tmp_path.iterdir()) == [path]
    resumed = make_sampler(path, resume=True)
    assert resumed.n_like == sampler.n_like
    assert resumed.log_z == sampler.log_z


def test_atomic_write_does_not_overwrite_racing_destination(tmp_path, monkeypatch):
    path = tmp_path / 'run.h5'
    sampler = make_sampler()
    sampler.run(n_like_max=32)
    write = sampler._write

    def concurrent_write(stream):
        write(stream)
        path.write_bytes(b'other writer')

    monkeypatch.setattr(sampler, '_write', concurrent_write)
    with pytest.raises(RuntimeError, match='already exists'):
        sampler.write(path)
    assert path.read_bytes() == b'other writer'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('discard', [False, True])
def test_incremental_matches_full_and_resumes(tmp_path, discard):
    path = tmp_path / 'incremental.h5'
    full = tmp_path / 'full.h5'
    sampler = make_sampler(path)
    sampler.run(f_live=0.4, n_eff=0, discard_exploration=discard)
    for shell in range(len(sampler.bounds)):
        sampler.add_samples(shell)
        sampler.write_shell_update(path, shell)
    sampler.write(full)
    with h5py.File(path) as left, h5py.File(full) as right:
        assert_hdf5_equal(left, right)
    resumed = make_sampler(path, resume=True)
    for instance in (sampler, resumed):
        instance.run(f_live=0.4, n_eff=600, discard_exploration=discard)
    for left, right in zip(sampler.posterior(), resumed.posterior()):
        np.testing.assert_array_equal(left, right)
    assert sampler.log_z == resumed.log_z
    assert sampler.rng.bit_generator.state == resumed.rng.bit_generator.state


@pytest.mark.parametrize('shape', [(0,), (0, 22)])
def test_append_rows_preserves_previous_views(shape):
    array = np.zeros(shape)
    old = []
    reference = array.copy()
    rng = np.random.default_rng(0)
    for size in [64, 512, 3, 2048, 64, 10, 70000, 5]:
        old.append((array, array.copy()))
        values = rng.normal(size=(size,) + shape[1:])
        array = _append_rows(array, values)
        reference = np.append(reference, values, axis=0)
        np.testing.assert_array_equal(array, reference)
        for previous, snapshot in old:
            np.testing.assert_array_equal(previous, snapshot)
    # Capacity growth has a bounded memory cost, independent of shell length.
    assert len(array.base)-len(array) <= 65536


def test_append_rows_structured_and_noncontiguous():
    dtype = [('value', 'f8'), ('flag', 'i8')]
    array = np.zeros(8, dtype=dtype)[::2]
    values = np.array([(3.2, 1), (7.1, 0)], dtype=dtype)
    np.testing.assert_array_equal(_append_rows(array, values),
                                  np.append(array, values, axis=0))
