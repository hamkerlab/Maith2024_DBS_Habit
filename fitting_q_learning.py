"""
based on:

Ricardo Vieira . "Fitting a Reinforcement Learning Model to Behavioral Data with PyMC". In: PyMC Examples. Ed. by PyMC Team. DOI: 10.5281/zenodo.5654871
url: https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html#estimating-the-learning-parameters-via-pymc

and

Learn PyMC & Bayesian modeling > Notebooks on core features > Model comparison
url: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html

"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def generate_data(rng, alpha_plus, alpha_minus, beta, n=100, p_r=None):
    if p_r is None:
        p_r = [0.4, 0.6]
    actions = np.zeros(n, dtype="int")
    rewards = np.zeros(n, dtype="int")
    Qs = np.zeros((n, 2))

    # Initialize Q table
    Q = np.array([0.5, 0.5])
    for i in range(n):
        # Apply the Softmax transformation
        exp_Q = np.exp(beta * (Q - np.max(Q)))
        prob_a = exp_Q / np.sum(exp_Q)

        # Simulate choice and reward
        a = rng.choice([0, 1], p=prob_a)
        r = rng.random() < p_r[a]

        # Update Q table
        if (r - Q[a]) > 0:
            Q[a] = Q[a] + alpha_plus * (r - Q[a])
        else:
            Q[a] = Q[a] + alpha_minus * (r - Q[a])

        # Store values
        actions[i] = a
        rewards[i] = r
        Qs[i] = Q.copy()

    return actions, rewards, Qs


def update_Q(action, reward, Qs, alpha_plus, alpha_minus):
    """
    This function updates the Q table according to the RL update rule.
    It will be called by pytensor.scan to do so recursevely, given the observed data and the alpha parameter
    This could have been replaced be the following lamba expression in the pytensor.scan fn argument:
        fn=lamba action, reward, Qs, alpha: pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
    """

    # Use pt.switch to select alpha based on the comparison
    alpha = pt.switch(reward > Qs[action], alpha_plus, alpha_minus)

    # Update Q-value using the selected alpha
    Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))

    return Qs


def update_Q_single(action, reward, Qs, alpha):
    """
    This function updates the Q table according to the RL update rule.
    It will be called by pytensor.scan to do so recursevely, given the observed data and the alpha parameter
    This could have been replaced be the following lamba expression in the pytensor.scan fn argument:
        fn=lamba action, reward, Qs, alpha: pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
    """

    Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))

    return Qs


def action_is_one_probs(alpha_plus, alpha_minus, beta, actions, rewards):
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Qs values
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, _ = pytensor.scan(
        fn=update_Q,
        sequences=[actions, rewards],
        outputs_info=[Qs],
        non_sequences=[alpha_plus, alpha_minus],
    )

    # Apply the sotfmax transformation
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Return the probabilities for the right action, in the original scale
    return pt.exp(logp_actions[:, 1])


def action_is_one_probs_single(alpha, beta, actions, rewards):
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Qs values
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, _ = pytensor.scan(
        fn=update_Q_single,
        sequences=[actions, rewards],
        outputs_info=[Qs],
        non_sequences=[alpha],
    )

    # Apply the sotfmax transformation
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Return the probabilities for the right action, in the original scale
    return pt.exp(logp_actions[:, 1])


if __name__ == "__main__":
    az.style.use("arviz-darkgrid")
    seed = 123
    rng = np.random.default_rng(seed)

    true_alpha_plus = 0.9
    true_alpha_minus = 0.1
    true_beta = 5
    n = 120
    actions, rewards, _ = generate_data(
        rng, true_alpha_plus, true_alpha_minus, true_beta, n
    )

    # model with single learning rate
    with pm.Model() as m_bernoulli_single:
        alpha = pm.Beta(name="alpha", alpha=1, beta=1)
        beta = pm.HalfNormal(name="beta", sigma=10)

        action_probs = action_is_one_probs_single(alpha, beta, actions, rewards)
        # actions are either 0 or 1 and you have the probability for each trial that the
        # action is 1, this can be described by a Bernoulli distribution
        # thus it can be estimated how likely the observed actions are given the model
        # from which the probabilities were derived
        like = pm.Bernoulli(name="like", p=action_probs, observed=actions[1:])

        trace_single = pm.sample(random_seed=rng)

    with m_bernoulli_single:
        pm.compute_log_likelihood(trace_single)

    # model with two learning rates
    with pm.Model() as m_bernoulli:
        alpha_plus = pm.Beta(name="alpha_plus", alpha=1, beta=1)
        alpha_minus = pm.Beta(name="alpha_minus", alpha=1, beta=1)
        beta = pm.HalfNormal(name="beta", sigma=10)

        action_probs = action_is_one_probs(
            alpha_plus, alpha_minus, beta, actions, rewards
        )
        # actions are either 0 or 1 and you have the probability for each trial that the
        # action is 1, this can be described by a Bernoulli distribution
        # thus it can be estimated how likely the observed actions are given the model
        # from which the probabilities were derived
        like = pm.Bernoulli(name="like", p=action_probs, observed=actions[1:])

        trace = pm.sample(random_seed=rng)

    with m_bernoulli:
        pm.compute_log_likelihood(trace)

    df_comp_loo = az.compare({"m_bernoulli_single": trace_single, "m_bernoulli": trace})
    print(df_comp_loo)

    az.plot_compare(df_comp_loo, insample_dev=False)
    az.plot_trace(data=trace)
    az.plot_posterior(
        data=trace, ref_val=[true_alpha_plus, true_alpha_minus, true_beta]
    )
    plt.show()
