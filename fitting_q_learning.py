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


def generate_data_q_learn(rng, alpha_plus, alpha_minus, beta, n=100, p=0.6):
    """
    This function generates data from a simple Q-learning model. It simulates a
    two-armed bandit task with actions 0 and 1, and reward probabilities P(R(a=1)) = p
    and P(R(a=0)) = 1-p. The Q-learning model uses softmax action selection and
    two learning rates, alpha_plus and alpha_minus, for positive and negative
    prediction errors, respectively.

    Args:
        rng (numpy.random.Generator):
            Random number generator.
        alpha_plus (float):
            Learning rate for positive prediction errors.
        alpha_minus (float):
            Learning rate for negative prediction errors.
        beta (float):
            Inverse temperature parameter.
        n (int):
            Number of trials.
        p (float):
            Probability of reward for the second action.

    Returns:
        actions (numpy.ndarray):
            Vector of actions.
        rewards (numpy.ndarray):
            Vector of rewards.
        Qs (numpy.ndarray):
            Matrix of Q-values (n x 2).
    """
    # reward probabilities
    prob_r = [1 - p, p]
    # init variables
    actions = np.zeros(n, dtype="int")
    rewards = np.zeros(n, dtype="int")
    Qs = np.zeros((n, 2))
    # init action values (Q-values)
    Q = np.array([0.5, 0.5])
    # loop over trials
    for i in range(n):
        # compute action probabilities using softmax
        exp_Q = np.exp(beta * (Q - np.max(Q)))
        prob_a = exp_Q / np.sum(exp_Q)
        # action selection and reward
        a = rng.choice([0, 1], p=prob_a)
        r = rng.random() < prob_r[a]
        # update Q-values
        if (r - Q[a]) > 0:
            Q[a] = Q[a] + alpha_plus * (r - Q[a])
        else:
            Q[a] = Q[a] + alpha_minus * (r - Q[a])
        # store values
        actions[i] = a
        rewards[i] = r
        Qs[i] = Q.copy()

    return actions, rewards, Qs


def update_Q_single(action, reward, Qs, alpha):
    """
    This fucniton is called by pytensor.scan.
    It updates the Q-values according to the Q-learning update rule using a single
    learning rate.

    Args:
        action (int):
            Action taken.
        reward (int):
            Reward received.
        Qs (pytensor.TensorVariable):
            Q-values.
        alpha (float):
            Learning rate.

    Returns:
        Qs (pytensor.TensorVariable):
            Updated Q-values.
    """

    Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))

    return Qs


def get_action_is_one_probs_single(alpha, beta, actions, rewards):
    """
    This function computes the probability of selecting action 1 for each trial using
    Q-learning with given parameters and the given actions and rewards. Q-learning
    uses a single learning rate.

    Args:
        alpha (pytensor.TensorVariable):
            Learning rate.
        beta (pytensor.TensorVariable):
            Inverse temperature parameter.
        actions (numpy.ndarray):
            Vector of actions.
        rewards (numpy.ndarray):
            Vector of rewards.

    Returns:
        probs (pytensor.TensorVariable):
            Vector of probabilities of selecting action 1 after each trial (except the
            last).
    """
    # Convert actions and rewards vectors to tensors
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Q-values for each trial (result = after each trial) using pytensor.scan
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, _ = pytensor.scan(
        fn=update_Q_single,
        sequences=[actions, rewards],
        outputs_info=[Qs],
        non_sequences=[alpha],
    )

    # Compute the log probabilities for each trial of the actions
    # Qs[-1] are the Q-values after the last trial which are not needed
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Return the probabilities of selecting action 1 after each trial
    return pt.exp(logp_actions[:, 1])


def update_Q_single_multi_subjects(action, reward, Qs, alpha):
    """
    This function is called by pytensor.scan. It updates the Q-values according to
    the Q-learning update rule using a vector of learning rates for multiple subjects.

    Args:
        action (n x 1 pytensor.TensorVariable):
            Action taken (one per subject).
        reward (n x 1 pytensor.TensorVariable):
            Reward received (one per subject).
        Qs (n x 2 pytensor.TensorVariable):
            Q-values (one set per subject).
        alpha (n x 1 pytensor.TensorVariable):
            Learning rates for each subject.

    Returns:
        Qs (n x 2 pytensor.TensorVariable):
            Updated Q-values (one set per subject).
    """

    # Update Q-values for each subject
    Qs = pt.set_subtensor(
        Qs[pt.arange(Qs.shape[0]), action],
        Qs[pt.arange(Qs.shape[0]), action]
        + alpha * (reward - Qs[pt.arange(Qs.shape[0]), action]),
    )

    return Qs


def get_action_is_one_probs_single_multi_subjects(alpha, beta, actions, rewards):
    """
    This function computes the probability of selecting action 1 for each trial using
    Q-learning for multiple subjects with different alpha and beta values.

    Args:
        alpha (n pytensor.TensorVariable):
            Learning rates (one per subject).
        beta (n pytensor.TensorVariable):
            Inverse temperature parameters (one per subject).
        actions (n x m numpy.ndarray):
            Matrix of actions (n subjects, m trials).
        rewards (n x m numpy.ndarray):
            Matrix of rewards (n subjects, m trials).

    Returns:
        probs (n x (m-1) pytensor.TensorVariable):
            Matrix of probabilities of selecting action 1 for each subject (except the last trial).
    """

    # Convert actions and rewards matrices to tensors
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    n_subjects = actions.shape[0]

    # Initialize Q-values (one set of [Q0, Q1] per subject)
    Qs = 0.5 * pt.ones((n_subjects, 2), dtype="float64")

    # Compute Q-values for each trial using pytensor.scan
    Qs, _ = pytensor.scan(
        fn=update_Q_single_multi_subjects,
        sequences=[actions.T, rewards.T],  # Transpose to iterate over trials
        outputs_info=[Qs],
        non_sequences=[alpha],
    )

    # Compute log probabilities for each trial (but exclude the last trial's Q-values)
    Qs = Qs[:-1] * beta[None, :, None]
    logp_actions = Qs - pt.logsumexp(Qs, axis=2, keepdims=True)

    # Return probabilities of selecting action 1 after each trial for each subject
    return pt.exp(logp_actions[:, :, 1]).T


def update_Q(action, reward, Qs, alpha_plus, alpha_minus):
    """
    This fucniton is called by pytensor.scan.
    It updates the Q-values according to the Q-learning update rule using two learning
    rates.

    Args:
        action (int):
            Action taken.
        reward (int):
            Reward received.
        Qs (pytensor.TensorVariable):
            Q-values.
        alpha_plus (float):
            Learning rate for positive prediction errors.
        alpha_minus (float):
            Learning rate for negative prediction errors.

    Returns:
        Qs (pytensor.TensorVariable):
            Updated Q-values.
    """

    # Use pt.switch to select alpha based on the comparison
    alpha = pt.switch(reward > Qs[action], alpha_plus, alpha_minus)

    # Update Q-value using the selected alpha
    Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))

    return Qs


def get_action_is_one_probs(alpha_plus, alpha_minus, beta, actions, rewards):
    """
    This function computes the probability of selecting action 1 for each trial using
    Q-learning with given parameters and the given actions and rewards. Q-learning
    uses two learning rates.

    Args:
        alpha_plus (pytensor.TensorVariable):
            Learning rate for positive prediction errors.
        alpha_minus (pytensor.TensorVariable):
            Learning rate for negative prediction errors.
        beta (pytensor.TensorVariable):
            Inverse temperature parameter.
        actions (numpy.ndarray):
            Vector of actions.
        rewards (numpy.ndarray):
            Vector of rewards.

    Returns:
        probs (pytensor.TensorVariable):
            Vector of probabilities of selecting action 1 after each trial (except the
            last).
    """
    # Convert actions and rewards vectors to tensors
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Q-values for each trial (result = after each trial) using pytensor.scan
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, _ = pytensor.scan(
        fn=update_Q,
        sequences=[actions, rewards],
        outputs_info=[Qs],
        non_sequences=[alpha_plus, alpha_minus],
    )

    # Compute the log probabilities for each trial of the actions
    # Qs[-1] are the Q-values after the last trial which are not needed
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Return the probabilities of selecting action 1 after each trial
    return pt.exp(logp_actions[:, 1])


def update_Q_multi_subjects(action, reward, Qs, alpha_plus, alpha_minus):
    """
    This function is called by pytensor.scan. It updates the Q-values according to
    the Q-learning update rule using a vector of learning rates for multiple subjects.

    Args:
        action (n x 1 pytensor.TensorVariable):
            Action taken (one per subject).
        reward (n x 1 pytensor.TensorVariable):
            Reward received (one per subject).
        Qs (n x 2 pytensor.TensorVariable):
            Q-values (one set per subject).
        alpha_plus (n x 1 pytensor.TensorVariable):
            Learning rates for positive prediction errors (one per subject).
        alpha_minus (n x 1 pytensor.TensorVariable):
            Learning rates for negative prediction errors (one per subject).

    Returns:
        Qs (n x 2 pytensor.TensorVariable):
            Updated Q-values (one set per subject).
    """

    # Use pt.switch to select alpha based on the comparison
    alpha = pt.switch(
        reward > Qs[pt.arange(Qs.shape[0]), action], alpha_plus, alpha_minus
    )

    # Update Q-values for each subject
    Qs = pt.set_subtensor(
        Qs[pt.arange(Qs.shape[0]), action],
        Qs[pt.arange(Qs.shape[0]), action]
        + alpha * (reward - Qs[pt.arange(Qs.shape[0]), action]),
    )

    return Qs


def get_action_is_one_probs_multi_subjects(
    alpha_plus, alpha_minus, beta, actions, rewards
):
    """
    This function computes the probability of selecting action 1 for each trial using
    Q-learning for multiple subjects with different alpha and beta values.

    Args:
        alpha_plus (n pytensor.TensorVariable):
            Learning rates for positive prediction errors (one per subject).
        alpha_minus (n pytensor.TensorVariable):
            Learning rates for negative prediction errors (one per subject).
        beta (n pytensor.TensorVariable):
            Inverse temperature parameters (one per subject).
        actions (n x m numpy.ndarray):
            Matrix of actions (n subjects, m trials).
        rewards (n x m numpy.ndarray):
            Matrix of rewards (n subjects, m trials).

    Returns:
        probs (n x (m-1) pytensor.TensorVariable):
            Matrix of probabilities of selecting action 1 for each subject (except the last trial).
    """

    # Convert actions and rewards matrices to tensors
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    n_subjects = actions.shape[0]

    # Initialize Q-values (one set of [Q0, Q1] per subject)
    Qs = 0.5 * pt.ones((n_subjects, 2), dtype="float64")

    # Compute Q-values for each trial using pytensor.scan
    Qs, _ = pytensor.scan(
        fn=update_Q_multi_subjects,
        sequences=[actions.T, rewards.T],  # Transpose to iterate over trials
        outputs_info=[Qs],
        non_sequences=[alpha_plus, alpha_minus],
    )

    # Compute log probabilities for each trial (but exclude the last trial's Q-values)
    Qs = Qs[:-1] * beta[None, :, None]
    logp_actions = Qs - pt.logsumexp(Qs, axis=2, keepdims=True)

    # Return probabilities of selecting action 1 after each trial for each subject
    return pt.exp(logp_actions[:, :, 1]).T


if __name__ == "__main__":
    az.style.use("arviz-darkgrid")
    seed = 123
    rng = np.random.default_rng(seed)

    # generate data using the true parameters
    true_alpha_plus = 0.9
    true_alpha_minus = 0.1
    true_beta = 5
    n = 121
    actions, rewards, _ = generate_data_q_learn(
        rng, true_alpha_plus, true_alpha_minus, true_beta, n
    )

    # # tests for multiple subjects functions
    # actions = np.array([1, 1, 1, 1, 1, 1])
    # rewards = np.array([0, 0, 0, 0, 0, 0])

    # actions2 = np.array([1, 1, 1, 1, 1, 1])
    # rewards2 = np.array([1, 1, 1, 1, 1, 1])

    # actions3 = np.array([0, 0, 0, 0, 0, 0])
    # rewards3 = np.array([1, 1, 1, 1, 1, 1])

    # actions_comb = np.stack([actions, actions2, actions3], axis=0)
    # rewards_comb = np.stack([rewards, rewards2, rewards3], axis=0)

    # # test single function for sample 1
    # print("function with single learning rates:")

    # alpha = pt.dscalar("alpha")
    # beta = pt.dscalar("beta")

    # ret = get_action_is_one_probs_single(alpha, beta, actions, rewards)

    # # compile the function
    # f = pytensor.function([alpha, beta], ret)

    # # test the function
    # ret = f(0.5, 5.0)

    # print(actions[1:])
    # print(rewards[1:])
    # print(ret)
    # print("\n")

    # # test single function for sample 2

    # alpha = pt.dscalar("alpha")
    # beta = pt.dscalar("beta")

    # ret = get_action_is_one_probs_single(alpha, beta, actions2, rewards2)

    # # compile the function
    # f = pytensor.function([alpha, beta], ret)

    # # test the function
    # ret = f(0.5, 5.0)

    # print(actions2[1:])
    # print(rewards2[1:])
    # print(ret)
    # print("\n")

    # # test multy single function for both samples

    # alpha = pt.vector("alpha")
    # beta = pt.vector("beta")

    # ret = get_action_is_one_probs_single_multi_subjects(
    #     alpha, beta, actions_comb, rewards_comb
    # )

    # # compile the function
    # f = pytensor.function([alpha, beta], ret)

    # # test the function
    # ret = f([0.5, 0.5, 0.5], [5.0, 5.0, 5.0])

    # print(actions_comb[:, 1:])
    # print(rewards_comb[:, 1:])
    # print(ret)
    # print("\n\n")

    # # test two function for sample 1
    # print("function with two learning rates:")

    # alpha_plus = pt.dscalar("alpha_plus")
    # alpha_minus = pt.dscalar("alpha_minus")
    # beta = pt.dscalar("beta")

    # ret = get_action_is_one_probs(alpha_plus, alpha_minus, beta, actions, rewards)

    # # compile the function
    # f = pytensor.function([alpha_plus, alpha_minus, beta], ret)

    # # test the function
    # ret = f(0.9, 0.1, 5.0)

    # print(actions[1:])
    # print(rewards[1:])
    # print(ret)
    # print("\n")

    # # test two function for sample 2

    # alpha_plus = pt.dscalar("alpha_plus")
    # alpha_minus = pt.dscalar("alpha_minus")
    # beta = pt.dscalar("beta")

    # ret = get_action_is_one_probs(alpha_plus, alpha_minus, beta, actions2, rewards2)

    # # compile the function
    # f = pytensor.function([alpha_plus, alpha_minus, beta], ret)

    # # test the function
    # ret = f(0.9, 0.1, 5.0)

    # print(actions2[1:])
    # print(rewards2[1:])
    # print(ret)
    # print("\n")

    # # test multy two function for both samples

    # alpha_plus = pt.vector("alpha_plus")
    # alpha_minus = pt.vector("alpha_minus")
    # beta = pt.vector("beta")

    # ret = get_action_is_one_probs_multi_subjects(
    #     alpha_plus, alpha_minus, beta, actions_comb, rewards_comb
    # )

    # # compile the function
    # f = pytensor.function([alpha_plus, alpha_minus, beta], ret)

    # # test the function
    # ret = f([0.9, 0.9, 0.9], [0.1, 0.1, 0.1], [5.0, 5.0, 5.0])

    # print(actions_comb[:, 1:])
    # print(rewards_comb[:, 1:])
    # print(ret)

    # quit()

    # model with single learning rate
    with pm.Model() as m_bernoulli_single:
        # prior for the learning rate (alpha) and the inverse temperature (beta)
        alpha = pm.Beta(name="alpha", alpha=1, beta=1)
        beta = pm.HalfNormal(name="beta", sigma=10)
        # compute the probability of selecting action 1 after each trial
        action_is_one_probs = pm.Deterministic(
            "action_is_one_probs",
            get_action_is_one_probs_single(alpha, beta, actions, rewards),
        )
        # observed data (actions are either 0 or 1) can be modeled as Bernoulli
        # likelihood with the computed probabilities
        like = pm.Bernoulli(name="like", p=action_is_one_probs, observed=actions[1:])
        # sample from the posterior
        trace_single = pm.sample(random_seed=rng)
        # compute the log likelihood of the model
        pm.compute_log_likelihood(trace_single)

    # model with two learning rates
    with pm.Model() as m_bernoulli:
        alpha_plus = pm.Beta(name="alpha_plus", alpha=1, beta=1)
        alpha_minus = pm.Beta(name="alpha_minus", alpha=1, beta=1)
        beta = pm.HalfNormal(name="beta", sigma=10)
        action_is_one_probs = pm.Deterministic(
            "action_is_one_probs",
            get_action_is_one_probs(alpha_plus, alpha_minus, beta, actions, rewards),
        )
        like = pm.Bernoulli(name="like", p=action_is_one_probs, observed=actions[1:])
        trace = pm.sample(random_seed=rng)
        pm.compute_log_likelihood(trace)

    # model comparison using LOO (Leave-One-Out cross-validation)
    df_comp_loo = az.compare({"m_bernoulli_single": trace_single, "m_bernoulli": trace})
    print(df_comp_loo)

    # plot results of the model comparison
    az.plot_compare(df_comp_loo, insample_dev=False)

    # plot the posterior distributions of the parameters for the model with two learning
    # rates
    az.plot_trace(
        data=trace,
        var_names=["alpha_plus", "alpha_minus", "beta"],
    )
    az.plot_posterior(
        data=trace,
        var_names=["alpha_plus", "alpha_minus", "beta"],
        ref_val=[true_alpha_plus, true_alpha_minus, true_beta],
    )
    # plot the model
    pm.model_to_graphviz(m_bernoulli).render("model_bernoulli")
    plt.show()
