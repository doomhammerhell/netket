# Copyright 2026 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from netket._src.stats.local_estimators import LocalEstimatorsBatch
from netket.jax import HashablePartial
from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable
from netket.stats import Stats
from netket.utils.numbers import is_scalar
from netket.vqs import (
    FullSumState,
    MCState,
    expect,
    get_local_kernel,
    get_local_kernel_arguments,
)
from netket.vqs.mc.common import local_estimators


def _vscore_combinator(trace_diagonal: float, N: int, mu):
    return N * (mu[1] - mu[0] ** 2) / (mu[0] - trace_diagonal) ** 2


def _vscore_n_dof(hilbert) -> int:
    r"""Estimate the V-score normalisation :math:`N` (number of degrees of
    freedom) for a given Hilbert space, following arXiv:2302.04919.

    For spin, qubit and bosonic lattice models this is the number of sites/modes
    (``hilbert.size``). For fermionic systems the V-score is normalised by the
    total particle number, so we use ``SpinOrbitalFermions.n_fermions`` instead
    of ``hilbert.size`` (which counts spin-orbitals).
    """
    from netket.hilbert import SpinOrbitalFermions

    if isinstance(hilbert, SpinOrbitalFermions):
        if hilbert.n_fermions is None:
            raise ValueError(
                "Cannot infer the V-score normalisation `N` for a "
                "SpinOrbitalFermions space with unfixed particle number. "
                "Pass `N=...` explicitly to VScore: the V-score is normalised by "
                "the total particle number (see arXiv:2302.04919)."
            )
        return hilbert.n_fermions
    return hilbert.size


class VScore(AbstractObservable):
    r"""
    Observable computing the V-score of a quantum operator :math:`H`, as defined
    in `Wu et al., arXiv:2302.04919 <https://arxiv.org/abs/2302.04919>`_:

    .. math::

        V_{\mathrm{score}} =
        \frac{N\,\mathrm{Var}(H)}{(\langle H \rangle - E_\infty)^2}
        =
        \frac{N\,(\langle H^2 \rangle - \langle H \rangle^2)}
        {(\langle H \rangle - E_\infty)^2}.

    The ``trace_diagonal`` parameter is the infinite-temperature reference energy
    :math:`E_\infty := \mathrm{Tr}(H) / \dim(\mathcal{H})` entering the denominator
    (it is :math:`0` for traceless Hamiltonians, e.g. spin models built from Pauli
    operators).

    The prefactor :math:`N` is the number of degrees of freedom. By default it is
    inferred from the Hilbert space:

    * for spin, qubit and bosonic lattice models, ``N = hilbert.size`` (number of
      sites/modes);
    * for fermionic systems (:class:`~netket.hilbert.SpinOrbitalFermions`),
      ``N`` is the total particle number ``hilbert.n_fermions``.

    For composite or exotic Hilbert spaces, or to use a different convention, pass
    ``N`` explicitly.
    """

    def __init__(
        self,
        operator: AbstractOperator,
        *,
        trace_diagonal: float,
        N: int | None = None,
    ):
        super().__init__(operator.hilbert)

        trace_diagonal = jnp.asarray(trace_diagonal)
        if (not is_scalar(trace_diagonal)) or jnp.iscomplexobj(trace_diagonal):
            raise TypeError("`trace_diagonal` should be a real scalar number.")

        self._operator = operator
        self._operator_squared = operator @ operator
        self._trace_diagonal = float(trace_diagonal)
        self._N = int(N) if N is not None else _vscore_n_dof(operator.hilbert)

    @property
    def operator(self) -> AbstractOperator:
        return self._operator

    @property
    def operator_squared(self) -> AbstractOperator:
        return self._operator_squared

    @property
    def trace_diagonal(self) -> float:
        return self._trace_diagonal

    @property
    def N(self) -> int:
        """The number of degrees of freedom normalising the V-score."""
        return self._N

    def __repr__(self):
        return (
            f"VScore(op={self.operator}, trace_diagonal={self.trace_diagonal}, "
            f"N={self.N})"
        )


@local_estimators.dispatch
def vscore_local_estimators(
    vstate: MCState, vscore_op: VScore, chunk_size: int | None
) -> LocalEstimatorsBatch:  # noqa: F811
    if vscore_op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    local_kernel = get_local_kernel(vstate, vscore_op.operator, chunk_size)
    local_kernel2 = get_local_kernel(vstate, vscore_op.operator_squared, chunk_size)

    sigma, args = get_local_kernel_arguments(vstate, vscore_op.operator)
    sigma, args2 = get_local_kernel_arguments(vstate, vscore_op.operator_squared)

    n_chains = sigma.shape[0]
    if jnp.ndim(sigma) != 2:
        sigma = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(
            sigma, 0, sigma.ndim - 1
        )

    if chunk_size is not None:
        local_kernel = partial(local_kernel, chunk_size=chunk_size)
        local_kernel2 = partial(local_kernel2, chunk_size=chunk_size)

    W = vstate.variables
    O_loc = local_kernel(vstate._apply_fun, W, sigma, args).real
    O2_loc = local_kernel2(vstate._apply_fun, W, sigma, args2).real

    data = jnp.stack([O_loc, O2_loc], axis=-1).reshape(n_chains, -1, 2)
    return LocalEstimatorsBatch(
        data=data,
        combinator=HashablePartial(
            _vscore_combinator, vscore_op.trace_diagonal, vscore_op.N
        ),
    )


@expect.dispatch
def expect(vstate: MCState, vscore_op: VScore, chunk_size: int | None):  # noqa: F811
    return local_estimators(vstate, vscore_op, chunk_size).to_stats()


@expect.dispatch
def expect(vstate: FullSumState, vscore_op: VScore):  # noqa: F811
    if vscore_op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    operator_mtrx = vscore_op.operator.to_dense()
    operator_squared_mtrx = vscore_op.operator_squared.to_dense()

    return _expect_vscore_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        operator_mtrx,
        operator_squared_mtrx,
        vscore_op.trace_diagonal,
        vscore_op.N,
    )


@partial(jax.jit, static_argnames=("afun",))
def _expect_vscore_fs(
    afun,
    params,
    model_state,
    sigma,
    operator_mtrx,
    operator_squared_mtrx,
    trace_diagonal,
    N,
):
    W = {"params": params, **model_state}

    state = jnp.exp(afun(W, sigma))
    state = state / jnp.linalg.norm(state)

    E_mean = (state.conj() @ (operator_mtrx @ state)).real
    E2_mean = (state.conj() @ (operator_squared_mtrx @ state)).real
    vscore = N * (E2_mean - E_mean**2) / (E_mean - trace_diagonal) ** 2

    return Stats(mean=vscore, error_of_mean=0.0, variance=0.0)
