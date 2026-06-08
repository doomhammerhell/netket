import builtins

import pytest

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk


@pytest.fixture
def block_import(monkeypatch, blocked_modules):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in blocked_modules:
            raise ImportError(f"Blocked module '{name}' was imported")
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("block_import")
@pytest.mark.parametrize("blocked_modules", [["haiku"]])
def test_jax_framework_works_without_haiku():
    def init(rng, in_shape):
        out_shape = (1,)
        pars = {"0": jnp.ones((1,))}
        return out_shape, pars

    def apply(pars, x, **_):
        return pars["0"] * jnp.sum(x)

    hi = nk.hilbert.Qubit(8)
    sampler = nk.sampler.MetropolisLocal(hi)
    nk.vqs.MCState(sampler, model=(init, apply))


def test_haiku_framework():
    pytest.importorskip("dm_haiku")
    import haiku as hk

    def apply(x):
        net = hk.Sequential(
            [
                hk.Linear(10, with_bias=False),
                jax.nn.relu,
                hk.Linear(1, with_bias=False),
            ]
        )
        return net(x)[..., 0]

    hi = nk.hilbert.Qubit(8)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model=hk.transform(apply))

    assert vstate.n_parameters == hi.size * 10 + 10

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)


def test_equinox_framework():
    pytest.importorskip("equinox")
    import equinox as eqx

    class SupportBatch(eqx.Module):
        submodule: eqx.Module

        def __init__(self, submodule):
            self.submodule = submodule

        def __call__(self, x, **kwargs):
            return jax.vmap(lambda x: self.submodule(x, **kwargs))(x)

    L = 8

    ma = SupportBatch(
        eqx.nn.MLP(
            in_size=L, out_size="scalar", width_size=8, depth=1, key=jax.random.key(1)
        )
    )

    hi = nk.hilbert.Qubit(L)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, ma)

    assert vstate.n_parameters == hi.size * (ma.submodule.width_size + 2) + 1

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)

    np.testing.assert_allclose(vstate.model(hi.all_states()), logpsi)


def _make_flax_model():
    from flax import linen as nn

    class FlaxModel(nn.Module):
        features: int = 4

        @nn.compact
        def __call__(self, x):
            return jnp.sum(nn.Dense(self.features)(x), axis=-1)

    return FlaxModel, FlaxModel()


def _make_nnx_model():
    from flax import nnx

    class NNXModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.linear = nnx.Linear(4, 4, rngs=rngs)

        def __call__(self, x):
            return jnp.sum(self.linear(x), axis=-1)

    return NNXModel, NNXModel(rngs=nnx.Rngs(0))


def _make_equinox_model():
    import equinox as eqx

    class EquinoxModel(eqx.Module):
        w: jax.Array

        def __init__(self, key):
            self.w = jax.random.normal(key, (4,))

        def __call__(self, x):
            return jnp.sum(x * self.w, axis=-1)

    return EquinoxModel, EquinoxModel(jax.random.key(0))


def _make_bound_flax_model():
    # A *bound* flax module is unbound on `wrap` (via `module.unbind()`), so the
    # static module is a genuine instance of the user class and the base default
    # `wrapped_model_class` recovers it.
    from flax import linen as nn

    class BoundFlaxModel(nn.Module):
        features: int = 4

        def setup(self):
            self.dense = nn.Dense(self.features)

        def __call__(self, x):
            return jnp.sum(self.dense(x), axis=-1)

    module = BoundFlaxModel()
    variables = module.init(jax.random.key(0), jnp.ones((2, 4)))
    return BoundFlaxModel, module.bind(variables)


@pytest.mark.parametrize(
    "factory_name, required_module",
    [
        ("_make_flax_model", "flax"),
        ("_make_bound_flax_model", "flax"),
        ("_make_nnx_model", "flax"),
        ("_make_equinox_model", "equinox"),
    ],
)
def test_wrapped_model_class(factory_name, required_module):
    # `wrapped_model_class` must recover the original user-defined model class
    # from the (possibly wrapped) static module, without unwrapping it (which
    # would require the variables).
    pytest.importorskip(required_module)

    cls, module = globals()[factory_name]()

    framework = nk.utils.model_frameworks.identify_framework(module)
    _variables, static_module = framework.wrap(module)

    # The static module may be a wrapper (nnx/equinox) or the module itself
    # (flax, bound flax), but in all cases we must recover the original class.
    assert framework.wrapped_model_class(static_module) is cls

    # And the same must hold for the static module actually stored in MCState.
    hi = nk.hilbert.Qubit(4)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, module)

    assert vstate._model_framework.wrapped_model_class(vstate._model) is cls


def test_equinox_framework_no_key_kwarg():
    # Regression test: a plain Equinox module whose `__call__` does not
    # declare a `key` argument must still work. Previously the wrapper
    # unconditionally forwarded `key=None`, raising a TypeError.
    pytest.importorskip("equinox")
    import equinox as eqx

    L = 8

    class LinearNet(eqx.Module):
        w: jax.Array

        def __init__(self, key):
            self.w = jax.random.normal(key, (L,))

        def __call__(self, x):
            return jnp.sum(x * self.w, axis=-1)

    ma = LinearNet(jax.random.key(0))

    hi = nk.hilbert.Qubit(L)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, ma)

    assert vstate.n_parameters == L

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)

    np.testing.assert_allclose(vstate.model(hi.all_states()), logpsi)


def test_nnx_framework():
    pytest.importorskip("flax")
    from flax import nnx

    class SimpleNNX(nnx.Module):
        def __init__(self, features, rngs: nnx.Rngs):
            self.linear = nnx.Linear(4, features, rngs=rngs)

        def __call__(self, x):
            return jnp.sum(self.linear(x), axis=-1)

        def hidden_features(self, x):
            return self.linear(x)

    # Create the NNX module
    module = SimpleNNX(features=8, rngs=nnx.Rngs(0))

    # Test maybe_wrap_module
    variables, wrapped_module = nk.utils.model_frameworks.maybe_wrap_module(module)

    # Check that wrapped module has apply method
    assert hasattr(wrapped_module, "apply")

    # Check that parameters are in 'params' key
    assert "params" in variables
    assert "model_state" in variables

    # Test the apply method works
    test_input = jnp.ones((2, 4))

    # Test default __call__ method
    output1 = wrapped_module.apply(variables, test_input)
    assert output1.shape == (2,)

    # Test that additional method can be called through apply interface
    output2 = wrapped_module.apply(variables, test_input, method="hidden_features")
    assert output2.shape == (2, 8)

    # Test that the module can also be called directly through attribute access
    hidden_method = getattr(wrapped_module, "hidden_features")
    output3 = hidden_method(variables, test_input)
    assert output3.shape == (2, 8)

    # Verify outputs are the same
    np.testing.assert_allclose(output2, output3)


def test_bound_linen_framework():
    pytest.importorskip("flax")
    from flax import linen as nn

    class SimpleLinen(nn.Module):
        features: int

        def setup(self):
            self.linear = nn.Dense(self.features)

        def __call__(self, x):
            return jnp.sum(self.linear(x), axis=-1)

        def hidden_features(self, x):
            return self.linear(x)

    module = SimpleLinen(features=8)
    variables = module.init(jax.random.key(0), jnp.ones((2, 4)))
    bound_module = module.bind(variables)

    wrapped_variables, wrapped_module = nk.utils.model_frameworks.maybe_wrap_module(
        bound_module
    )

    assert wrapped_module.scope is None
    assert "params" in wrapped_variables

    test_input = jnp.ones((2, 4))
    output1 = wrapped_module.apply(wrapped_variables, test_input)
    output2 = wrapped_module.apply(
        wrapped_variables, test_input, method=wrapped_module.hidden_features
    )

    assert output1.shape == (2,)
    assert output2.shape == (2, 8)

    hi = nk.hilbert.Qubit(4)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, bound_module)

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)
    assert vstate._model.scope is None
    assert vstate.model.scope is not None

    np.testing.assert_allclose(vstate.model(hi.all_states()), logpsi)
    np.testing.assert_allclose(
        vstate.model.hidden_features(hi.all_states()),
        vstate._model.apply(
            vstate.variables,
            hi.all_states(),
            method=vstate._model.hidden_features,
        ),
    )
