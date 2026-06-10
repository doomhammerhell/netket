""" """

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from netket.vqs.mc import MCState
from netket.sampler import MetropolisSampler
from netket.operator import AbstractOperator
from netket._src.stats.online_stats import online_statistics
from netket._src.stats.online_stats.accumulator_batch import OnlineStatsBatch


def _summary_error_and_scale(stats) -> tuple[float, float]:
    """Return scalar error and scale summaries for scalar or batched observables."""
    s = stats.get_stats()
    if isinstance(stats, OnlineStatsBatch):
        err = float(jnp.max(jnp.abs(s.error_of_mean)))
        scale = float(jnp.max(jnp.abs(s.mean)))
    else:
        err = abs(float(jnp.real(s.error_of_mean)))
        scale = abs(float(jnp.real(s.mean)))
    return err, scale


def _tolerance(scale: float, atol: float | None, rtol: float | None) -> float:
    """Combined tolerance ``atol + rtol * scale``, NumPy-style.

    A missing tolerance is treated as 0, so ``atol`` acts as an absolute
    floor when the mean is close to zero.
    """
    return (atol or 0.0) + (rtol or 0.0) * scale


def _check_not_converged(stats, atol: float | None, rtol: float | None) -> bool:
    """Return True if the stats have not yet met the requested tolerance."""
    err, scale = _summary_error_and_scale(stats)
    return err > _tolerance(scale, atol, rtol)


def _format_postfix(stats, atol: float | None, rtol: float | None) -> dict:
    """Build the tqdm postfix dict from current stats and tolerances."""
    err, scale = _summary_error_and_scale(stats)
    return {
        "err": f"{err:.4g}",
        "tol": f"{_tolerance(scale, atol, rtol):.4g}",
    }


def _accumulate_stats(state, op_leaves, active, old_stats, *, max_lag):
    active = set(active)
    result = []
    for i, (op, old) in enumerate(zip(op_leaves, old_stats)):
        if i not in active:
            result.append(old)
            continue
        le = state.local_estimators(op)
        jax.block_until_ready(le)
        result.append(online_statistics(le, old_estimator=old, max_lag=max_lag))
    return result


def expect_to_precision(
    state: MCState,
    op: AbstractOperator,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    max_iter: int = 10_000,
    max_lag: int = 64,
    verbose: bool = True,
):
    """
    Sample until the estimated standard error of the mean meets the requested
    tolerance(s).

    This uses NetKet's online_statistics to update estimates incrementally.

    At least one of ``atol`` or ``rtol`` must be specified. Sampling stops when

    .. code-block:: python

        error_of_mean <= atol + rtol * |mean|

    with a missing tolerance treated as 0 (NumPy convention): with only
    ``atol`` the criterion is absolute, with only ``rtol`` it is relative,
    and with both, ``atol`` acts as an absolute floor that keeps the
    relative criterion well-behaved when the mean is close to zero.

    Args:
        state: The MC state to sample from.
        op: The operator whose expectation value is estimated.
        atol: Desired absolute standard error of the mean.
        rtol: Desired relative standard error of the mean.
        max_iter: Maximum number of sampling iterations.
        max_lag: Max lag used for autocorrelation estimation.
        verbose: Whether to show a progress bar.
    """
    if atol is None and rtol is None:
        raise ValueError("At least one of 'atol' or 'rtol' must be specified.")
    if atol is not None and atol <= 0:
        raise ValueError("atol must be > 0.")
    if rtol is not None and rtol <= 0:
        raise ValueError("rtol must be > 0.")

    if not isinstance(state.sampler, MetropolisSampler):
        raise ValueError("Only works with MetropolisSampler.")

    _is_rank0 = jax.process_index() == 0

    # Flatten the operator pytree once; all loop internals work on plain lists.
    op_leaves, treedef = jax.tree.flatten(
        op, is_leaf=lambda x: isinstance(x, AbstractOperator)
    )

    state.sample()
    stats_list = _accumulate_stats(
        state,
        op_leaves,
        range(len(op_leaves)),
        [None] * len(op_leaves),
        max_lag=max_lag,
    )
    active = [
        i
        for i in range(len(op_leaves))
        if _check_not_converged(stats_list[i], atol, rtol)
    ]

    it = 0
    with tqdm(
        total=max_iter,
        desc="Sampling",
        unit="iter",
        disable=not verbose or not _is_rank0,
    ) as pbar:
        pbar.set_postfix(
            _format_postfix(stats_list[active[0] if active else 0], atol, rtol)
        )
        try:
            while active and it < max_iter:
                state.sample(n_discard_per_chain=0)
                stats_list = _accumulate_stats(
                    state, op_leaves, active, stats_list, max_lag=max_lag
                )
                active = [
                    i for i in active if _check_not_converged(stats_list[i], atol, rtol)
                ]

                pbar.set_postfix(
                    _format_postfix(stats_list[active[0] if active else 0], atol, rtol)
                )
                pbar.update(1)
                it += 1
        except KeyboardInterrupt:
            if _is_rank0:
                pbar.write("  Early termination requested by user.")

        if verbose and _is_rank0:
            if it >= max_iter:
                pbar.write("  Reached max_iter before target precision.")
            err, _ = _summary_error_and_scale(stats_list[0])
            pbar.write(f"  [done] error = {err:g}")

    return treedef.unflatten(stats_list)
