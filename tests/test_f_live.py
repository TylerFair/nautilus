"""Incremental estimates must preserve the exact exploration stop."""
import numpy as np
import pytest

from nautilus import Sampler


def test_incremental_shell_cache():
    s = Sampler(lambda x: x, lambda x: 0., n_dim=2, n_live=7, backend='numpy')
    rng = np.random.default_rng(4)
    s.log_l = [rng.normal(size=n) for n in (40, 30, 0)]
    s.shell_n = np.array([40, 30, 0])
    s.shell_log_v = np.array([-1., -3., -np.inf])
    for change in ('initial', 'batch', 'volumes', 'remove', 'live'):
        old = getattr(s, '_f_live_shell_cache', (None, {}))[1]
        if change == 'batch':
            s.log_l[1] = np.append(s.log_l[1], rng.normal(size=5))
            s.shell_n[1] += 5
        elif change == 'volumes':
            s.shell_log_v[0] -= 1
        elif change == 'remove':
            s.log_l.pop()
            s.shell_n, s.shell_log_v = s.shell_n[:-1], s.shell_log_v[:-1]
        elif change == 'live':
            s.n_live = 3
        np.testing.assert_allclose(s._estimate_f_live(), s.f_live, rtol=1e-14)
        if old and change != 'live':
            assert s._f_live_shell_cache[1][id(s.log_l[0])] is old[id(s.log_l[0])]
        assert set(s._f_live_shell_cache[1]) == {id(a) for a in s.log_l if len(a)}


def test_ties_cannot_hide_exact_stop():
    s = Sampler(lambda x: x, lambda x: 0., n_dim=2, n_live=3, backend='numpy')
    # Tie ordering varies by NumPy version. Exercise a fast overestimate well
    # outside the factor-two confirmation window when such an ordering occurs.
    for n in (4, 7, 10, 50, 100):
        s.log_l = [np.zeros(n), np.zeros(n)]
        s.shell_n = np.array([n, n])
        for volumes in ([0., -30.], [-30., 0.]):
            s.shell_log_v = np.array(volumes)
            exact = s.f_live
            threshold = 1.1 * exact
            if s._estimate_f_live() > 2 * threshold:
                assert s._estimate_f_live(threshold) == exact
                return
    pytest.skip('This NumPy tie ordering has no overestimate in these examples')


@pytest.mark.parametrize('kind', ['gaussian', 'fp32', 'plateau'])
@pytest.mark.parametrize('seed', [0, 5])
def test_exact_stopping_and_posterior(kind, seed, monkeypatch, capsys):
    def likelihood(x):
        value = -np.sum((x - .5)**2) / .02
        if kind == 'fp32':
            return np.float32(value)
        return np.floor(value / 2) * 2 if kind == 'plateau' else value

    kwargs = dict(n_dim=3, n_live=80, n_networks=1, seed=seed)
    exact = Sampler(lambda x: x, likelihood, **kwargs)
    fast = Sampler(lambda x: x, likelihood, **kwargs)
    monkeypatch.setattr(exact, '_estimate_f_live', lambda threshold: exact.f_live)
    original = fast.print_status

    def status(*args, **kwargs):
        if not kwargs.get('header', False):
            assert 'f_live' in kwargs  # Progress never re-reads the property.
        original(*args, **kwargs)

    monkeypatch.setattr(fast, 'print_status', status)
    for s in (exact, fast):
        s.run(f_live=.2, n_eff=700, verbose=True)
    assert fast.n_like == exact.n_like
    np.testing.assert_array_equal(fast.shell_end_exp, exact.shell_end_exp)
    assert fast.log_z == exact.log_z
    assert fast.rng.bit_generator.state == exact.rng.bit_generator.state
    for a, b in zip(fast.posterior(), exact.posterior()):
        np.testing.assert_array_equal(a, b)


def test_cached_status_does_not_read_f_live(monkeypatch, capsys):
    s = Sampler(lambda x: x, lambda x: 0., n_dim=2, backend='numpy')
    monkeypatch.setattr(Sampler, 'f_live', property(lambda self: pytest.fail('read f_live')))
    s.print_status('Sampling', f_live=.3)
    assert '0.3000' in capsys.readouterr().out
