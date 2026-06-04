"""Tests for correlation function observables."""

import pytest
import numpy as np

import netket as nk


def _make_vstate(N=4, n_samples=512, exact=True, param_dtype=float):
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    ma = nk.models.RBM(alpha=1, param_dtype=param_dtype)
    if exact:
        sa = nk.sampler.ExactSampler(hilbert=hi)
    else:
        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
    return nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples), hi


class TestConnectedCorrelator:
    def test_agrees_with_product_expect(self):
        """<A B>_c matches vs.expect(A@B).mean - vs.expect(A).mean * vs.expect(B).mean."""
        vs, hi = _make_vstate()
        A = nk.operator.spin.sigmax(hi, 0)
        B = nk.operator.spin.sigmax(hi, 1)

        obs = nk.observable.ConnectedCorrelator(A, B)
        result = vs.expect(obs)

        AB_mean = vs.expect(A @ B).mean
        A_mean = vs.expect(A).mean
        B_mean = vs.expect(B).mean
        expected = AB_mean - A_mean * B_mean

        np.testing.assert_allclose(result.mean, expected, atol=1e-6)

    def test_sz_sz_connected(self):
        """<Z_i Z_j>_c on product state should vanish (uncorrelated)."""
        vs, hi = _make_vstate()
        Z0 = nk.operator.spin.sigmaz(hi, 0)
        Z1 = nk.operator.spin.sigmaz(hi, 1)
        obs = nk.observable.ConnectedCorrelator(Z0, Z1)
        result = vs.expect(obs)
        assert hasattr(result, "mean")

    @pytest.mark.parametrize("op", ["sigmaz", "sigmax"])
    def test_complex_amplitude_ansatz(self, op):
        """Correlator works on complex-amplitude ansätze (delta method needs real inputs).

        Complex models (e.g. ViT) and k=0-projected states produce complex-dtyped
        local estimators; the connected correlator must take the real part of the
        Hermitian channels rather than crashing in ``jax.jacfwd``.
        """
        vs, hi = _make_vstate(n_samples=4096, param_dtype=complex)
        make = getattr(nk.operator.spin, op)
        A = make(hi, 0)
        B = make(hi, 2)

        obs = nk.observable.ConnectedCorrelator(A, B)
        result = vs.expect(obs)

        expected = (
            vs.expect(A @ B).mean - vs.expect(A).mean * vs.expect(B).mean
        ).real
        assert np.isrealobj(result.mean)
        np.testing.assert_allclose(result.mean, expected, atol=1e-6)

    def test_hilbert_mismatch_raises(self):
        hi1 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi2 = nk.hilbert.Spin(s=1 / 2, N=3)
        A = nk.operator.spin.sigmax(hi1, 0)
        B = nk.operator.spin.sigmax(hi2, 0)
        with pytest.raises(ValueError, match="same Hilbert space"):
            nk.observable.ConnectedCorrelator(A, B)

    def test_local_estimators_returns_batch(self):
        vs, hi = _make_vstate()
        A = nk.operator.spin.sigmax(hi, 0)
        B = nk.operator.spin.sigmax(hi, 1)
        obs = nk.observable.ConnectedCorrelator(A, B)

        from netket._src.stats.local_estimators import LocalEstimatorsBatch

        le = vs.local_estimators(obs)
        assert isinstance(le, LocalEstimatorsBatch)
        assert le.n_channels == 3  # L_A, L_B, L_{AB}
