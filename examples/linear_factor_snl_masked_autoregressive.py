"""
SLCP example from [1] using SNL and masked coupling bijections or surjections
"""

from absl import logging
import argparse
from functools import partial

import distrax
import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from jax import random
from jax import vmap
from surjectors import Chain, TransformedDistribution
from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.bijectors.permutation import Permutation
from surjectors.conditioners import MADE, mlp_conditioner
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (  # type: ignore # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)
from surjectors.util import unstack

from sbijax import SNL
from sbijax.mcmc.slice import sample_with_slice


logging.set_verbosity(logging.DEBUG)

seed = random.PRNGKey(23)

D = 10  # Dimension of data vector
L = 5  # Dimension of latent variable

"""
Hardcode mu_0, Sigma_0, W, mu, and Psi for now.

Prior: p(z) = N(z | mu_0, Sigma_0)
Likelihood: N(x | Wz + mu, Psi)

Only z is learnable.
"""
mu_0 = jnp.zeros((L,))
Sigma_0 = 0.2 * jnp.eye(L)
W = jax.random.uniform(seed, shape=(D, L), minval=-3, maxval=3)
mu = jax.random.uniform(seed, shape=(D,), minval=-3, maxval=3)
Psi = 0.2 * jnp.eye(D)


def prior_model_fns():
    """
    Prior: p(z) = N(z | mu_0, Sigma_0)

    mu_0 and Sigma_0 are hardcoded
    """
    p = distrax.MultivariateNormalFullCovariance(mu_0, Sigma_0)
    return p.sample, p.log_prob


def likelihood_fn(theta, x):
    """
    Likelihood: N(x | Wz + mu, Psi)

    x is the data, z is the learnable parameters theta without any transformation

    z is theta
    z has shape [batch_size, 5]
    expected shape of loc [batch_size, 10]
    """
    z = theta

    loc = W @ z + mu
    p = distrax.MultivariateNormalFullCovariance(loc, Psi)
    return p.log_prob(x)


def simulator_fn(seed, theta):
    """
    Simulator function takes parameters theta and returns a sampled data

    z is theta
    z has shape [batch_size, 5]
    W has shape [10, 5]
    mu has shape [10]
    expected shape of mu [batch_size, 10]
    """

    z = theta
    assert len(z.shape) == 2

    loc = jnp.swapaxes(W @ z.T + mu.reshape([-1, 1]), 0, 1)
    p = distrax.MultivariateNormalFullCovariance(loc, Psi)

    x = p.sample(seed=random.split(seed)[0])

    return x


def make_model(dim, use_surjectors):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _decoder_fn(n_dim):
        decoder_net = mlp_conditioner(
            [50, n_dim * 2],
            w_init=hk.initializers.TruncatedNormal(stddev=0.001),
        )

        def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return distrax.Independent(
                distrax.Normal(mu, jnp.exp(log_scale)), 1
            )

        return _fn

    def _flow(method, **kwargs):
        layers = []
        n_dimension = dim
        order = jnp.arange(n_dimension)
        for i in range(5):
            if i == 2 and use_surjectors:
                n_latent = 6
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_latent,
                    _decoder_fn(n_dimension - n_latent),
                    conditioner=MADE(
                        n_latent,
                        [50, n_latent * 2],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                n_dimension = n_latent
                order = order[::-1]
                order = order[:n_dimension] - jnp.min(order[:n_dimension])
            else:
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        n_dimension,
                        [50, n_dimension * 2],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.001),
                        b_init=jnp.zeros,
                        activation=jax.nn.tanh,
                    ),
                )
                order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dimension), jnp.ones(n_dimension)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def generate_observations(
    n_observations: int, expected_z: jnp.array, noise_level: float = 1.0
) -> jnp.array:
    noise = distrax.Uniform(-noise_level, noise_level).sample(
        seed=random.split(seed)[0], sample_shape=(n_observations, D)
    )
    y_observed = W @ expected_z + mu + noise

    return y_observed


def run(use_surjectors):
    len_theta = L
    n_observations = 4
    expected_z = jnp.array([-4.0, 3.5, 2.7, -0.8, 1.2])

    y_observed = generate_observations(n_observations, expected_z, 2.0)

    prior_simulator_fn, prior_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_fn), simulator_fn

    def log_density_fn(theta, y):
        prior_lp = prior_fn(theta)
        likelihood_lp = likelihood_fn(theta, y)

        lp = jnp.sum(prior_lp) + jnp.sum(likelihood_lp)
        return lp

    log_density_partial = partial(log_density_fn, y=y_observed)

    def log_density(x):
        return vmap(log_density_partial)(x)

    model = make_model(y_observed.shape[1], use_surjectors)
    snl = SNL(fns, model)
    optimizer = optax.adam(3e-3)
    params, info = snl.fit(
        random.PRNGKey(23),
        y_observed,
        optimizer,
        n_rounds=2,
        n_samples=100,
        max_n_iter=100,
        n_warmup=25,
        batch_size=64,
        n_early_stopping_patience=5,
        sampler="slice",
    )

    slice_samples = sample_with_slice(
        hk.PRNGSequence(12), log_density, 4, 5000, 2500, prior_simulator_fn
    )
    slice_samples = slice_samples.reshape(-1, len_theta)
    snl_samples, _ = snl.sample_posterior(params, 4, 5000, 2500)

    g = sns.PairGrid(pd.DataFrame(slice_samples))
    g.map_upper(
        sns.scatterplot, color="darkgrey", marker=".", edgecolor=None, s=2
    )
    g.map_diag(plt.hist, color="darkgrey")
    for ax in g.axes.flatten():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    g.fig.set_figheight(5)
    g.fig.set_figwidth(5)
    plt.show(block=False)

    g = sns.PairGrid(pd.DataFrame(snl_samples))
    g.map_upper(
        sns.scatterplot, color="darkblue", marker=".", edgecolor=None, s=2
    )
    g.map_diag(plt.hist, color="darkblue")
    for ax in g.axes.flatten():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    g.fig.set_figheight(5)
    g.fig.set_figwidth(5)
    plt.show(block=False)

    fig, axes = plt.subplots(len_theta, 2)
    for i in range(len_theta):
        sns.histplot(slice_samples[:, i], color="darkgrey", ax=axes[i, 0])
        sns.histplot(snl_samples[:, i], color="darkblue", ax=axes[i, 1])
        axes[i, 0].set_title(rf"Sampled posterior $\theta_{i}$")
        axes[i, 1].set_title(rf"Approximated posterior $\theta_{i}$")
        for j in range(2):
            axes[i, j].set_xlim(-5, 5)
    sns.despine()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-surjectors", action="store_true", default=True)
    args = parser.parse_args()
    run(args.use_surjectors)
