"""Do not evaluate more bounds after all proposed points are excluded."""

import numpy as np

from nautilus import Sampler


class ProposalBound:
    def __init__(self, rng):
        self.rng = rng
        self.calls = 0

    def sample(self, count, pool=None):
        self.calls += 1
        values = self.rng.random((count, 2))
        values[:, 0] = 0.9 if self.calls == 1 else 0.1
        return values


class ExclusionBound:
    def __init__(self, threshold):
        self.threshold = threshold
        self.calls = 0

    def contains(self, values):
        self.calls += 1
        return values[:, 0] > self.threshold


def test_fully_excluded_batch_skips_later_bounds_without_changing_draws():
    sampler = Sampler.__new__(Sampler)
    sampler.rng = np.random.default_rng(94)
    sampler.n_batch = 3
    sampler.pool_s = None
    sampler.bounds = [ProposalBound(sampler.rng),
                      ExclusionBound(0.5), ExclusionBound(0.8)]
    points, proposed = sampler.sample_shell(0)
    expected_rng = np.random.default_rng(94)
    expected_rng.random((3, 2))
    expected = expected_rng.random((3, 2))
    expected[:, 0] = 0.1
    np.testing.assert_array_equal(points, expected)
    assert sampler.rng.bit_generator.state == expected_rng.bit_generator.state
    assert proposed == 6
    assert sampler.bounds[1].calls == 2
    assert sampler.bounds[2].calls == 1


def test_last_bound_still_returns_transfer_indices_without_extra_draws():
    sampler = Sampler.__new__(Sampler)
    sampler.rng = np.random.default_rng(94)
    sampler.n_batch = 3
    sampler.pool_s = None
    sampler.bounds = [ProposalBound(sampler.rng)]
    points, proposed, transferred = sampler.sample_shell(-1, shell_t=np.array([]))
    expected_rng = np.random.default_rng(94)
    expected = expected_rng.random((3, 2))
    expected[:, 0] = 0.9
    np.testing.assert_array_equal(points, expected)
    assert sampler.rng.bit_generator.state == expected_rng.bit_generator.state
    assert proposed == 3
    assert len(transferred) == 0
