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
import os


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


def update_Q_double(action, reward, Qs, alpha_plus, alpha_minus):
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


def get_action_is_one_probs_double(alpha_plus, alpha_minus, beta, actions, rewards):
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
        fn=update_Q_double,
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


def pad_nan_on_third_dim(array):
    max_array_size = max(len(sub_arr) for row in array for sub_arr in row)

    # Pad each array to the maximum length with NaN values
    return np.array(
        [
            [
                np.concatenate(
                    [sub_arr, np.full(max_array_size - len(sub_arr), np.nan)]
                )
                for sub_arr in row
            ]
            for row in array
        ]
    )


def get_probabilities_single(coords, alpha, beta, actions_arr, rewards_arr):
    # for loop over dbs
    action_is_one_probs_list = []
    for dbs in coords["dbs"]:
        # for loop over subjects
        for subject in coords["subjects"]:
            # compute the probability of selecting action 1 after each trial
            action_is_one_probs_list.append(
                get_action_is_one_probs_single(
                    alpha[subject, dbs],
                    beta[subject, dbs],
                    actions_arr[dbs, subject],
                    rewards_arr[dbs, subject],
                )
            )
    action_is_one_probs_arr = pt.concatenate(action_is_one_probs_list)

    return action_is_one_probs_arr


def get_probabilities_double(
    coords, alpha_plus, alpha_minus, beta, actions_arr, rewards_arr
):
    # for loop over dbs
    action_is_one_probs_list = []
    for dbs in coords["dbs"]:
        # for loop over subjects
        for subject in coords["subjects"]:
            # compute the probability of selecting action 1 after each trial
            action_is_one_probs_list.append(
                get_action_is_one_probs_double(
                    alpha_plus[subject, dbs],
                    alpha_minus[subject, dbs],
                    beta[subject, dbs],
                    actions_arr[dbs, subject],
                    rewards_arr[dbs, subject],
                )
            )
    action_is_one_probs_arr = pt.concatenate(action_is_one_probs_list)

    return action_is_one_probs_arr


def my_Gamma(name, mu, sigma, dims=None):
    var = sigma**2
    return pm.Gamma(
        name,
        alpha=mu**2 / var,
        beta=mu / var,
        dims=dims,
    )


def analyze_model(
    name,
    model,
    save_folder,
    rng,
    true_alpha_plus_2d,
    true_alpha_minus_2d,
    true_beta_2d,
    chains,
    tune,
    draws,
    draws_prior,
):
    # plot model
    pm.model_to_graphviz(model).render(f"{save_folder}/{name}_model_plot")

    # sample the prior
    with model:
        idata = pm.sample_prior_predictive(draws=draws_prior, random_seed=rng)

    # visualize the prior samples distribution
    for var_name in idata.prior.data_vars:
        # skip deterministic variables
        if var_name == "action_is_one_probs":
            continue
        az.plot_density(
            idata,
            group="prior",
            var_names=[var_name],
        )
        plt.savefig(f"{save_folder}/{name}_prior_samples_{var_name}.png")
        plt.close("all")

    # sample the posterior
    with model:
        # sample from the posterior
        idata.extend(pm.sample(random_seed=rng, chains=chains, tune=tune, draws=draws))
        # compute the log likelihood of the model
        pm.compute_log_likelihood(idata)

    # plot the posterior distributions
    for var_name in idata.posterior.data_vars:
        # skip deterministic variables
        if var_name == "action_is_one_probs":
            continue
        if var_name == "alpha":
            ref_val = true_alpha_plus_2d.flatten().tolist()
        elif var_name == "alpha_plus":
            ref_val = true_alpha_plus_2d.flatten().tolist()
        elif var_name == "alpha_minus":
            ref_val = true_alpha_minus_2d.flatten().tolist()
        elif var_name == "beta":
            ref_val = true_beta_2d.flatten().tolist()
        else:
            ref_val = None
        az.plot_posterior(
            data=idata,
            var_names=[var_name],
            ref_val=ref_val,
        )
        plt.savefig(f"{save_folder}/{name}_posterior_{var_name}.png")
        plt.close("all")

    # plot the forest plot
    az.plot_forest(
        idata,
        var_names=(
            ["alpha", "beta"]
            if "alpha" in idata.posterior.data_vars
            else ["alpha_plus", "alpha_minus", "beta"]
        ),
        r_hat=True,
        combined=True,
        figsize=(6, 18),
    )
    plt.savefig(f"{save_folder}/{name}_forest_plot.png")
    plt.close("all")

    # print the summary of the model
    az.summary(idata).to_csv(f"{save_folder}/{name}_summary.csv")

    # save inference data
    idata.to_netcdf(f"{save_folder}/{name}_idata.nc")

    return idata


if __name__ == "__main__":
    save_folder = "results_fitting_q_learning"
    seed = 123
    tune = 7000
    draws = 15000
    draws_prior = 2000
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    az.style.use("arviz-darkgrid")
    rng = np.random.default_rng(seed)

    # generate data using the true parameters
    n_subjects = 14
    true_alpha = np.clip(rng.normal(loc=0.3, scale=0.05, size=n_subjects), 0.01, 1.0)
    true_alpha_plus = true_alpha
    true_alpha_minus = np.clip(true_alpha + 0.15, 0.01, 1.0)
    true_alpha_plus_2d = np.stack([true_alpha_plus, true_alpha_plus], axis=1)
    true_alpha_minus_2d = np.stack([true_alpha_minus, true_alpha_minus], axis=1)

    actions_arr = np.empty((2, n_subjects), dtype=object)
    rewards_arr = np.empty((2, n_subjects), dtype=object)
    observed_list = []
    true_beta_arr = np.clip(rng.normal(loc=1.5, scale=0.25, size=n_subjects), 0.5, 3.0)
    true_beta_2d = np.stack([true_beta_arr, true_beta_arr / 2], axis=1)

    # print true alpha and beta in text file
    with open(f"{save_folder}/true_alpha_beta.txt", "w") as f:
        f.write(f"True alpha plus:\n{true_alpha_plus_2d}\n\n")
        f.write(f"True alpha minus:\n{true_alpha_minus_2d}\n\n")
        f.write(f"True beta:\n{true_beta_2d}\n")

    for dbs in [0, 1]:
        for subject in range(n_subjects):

            n = 100 + rng.integers(-20, 20)

            actions, rewards, _ = generate_data_q_learn(
                rng,
                true_alpha_plus_2d[subject, dbs],
                true_alpha_minus_2d[subject, dbs],
                true_beta_2d[subject, dbs],
                n,
            )

            actions_arr[dbs, subject] = actions
            rewards_arr[dbs, subject] = rewards
            observed_list.append(actions[1:])

    observed_arr = np.concatenate(observed_list)

    coords = {
        "dbs": range(actions_arr.shape[0]),
        "subjects": range(actions_arr.shape[1]),
    }

    # model with single learning rate
    with pm.Model(coords=coords) as m_bernoulli_single:
        # observed data
        observed_data = pm.Data("observed_data", observed_arr)

        # Group-level mean and standard deviation for learning rate
        alpha_mean_global = pm.TruncatedNormal(
            "alpha_mean_global",
            mu=0.3,
            sigma=0.1,
            lower=0.1,
            upper=0.6,
        )
        alpha_sig_global = pm.Exponential("alpha_sig_global", lam=60)

        # Group-level mean and standard deviation for the change of learning rate by dbs
        alpha_dbs_effect_mean_global = pm.Normal(
            "alpha_dbs_effect_mean_global", mu=0.0, sigma=0.2
        )
        alpha_dbs_effect_sig_global = pm.Exponential(
            "alpha_dbs_effect_sig_global", lam=50
        )

        # subject-level learning rate
        alpha_subject = pm.TruncatedNormal(
            "alpha_subject",
            mu=alpha_mean_global,
            sigma=alpha_sig_global,
            lower=0,
            upper=1,
            dims="subjects",
        )

        # subject-level change of learning rate by dbs
        alpha_subject_dbs_effect = pm.TruncatedNormal(
            "alpha_subject_dbs_effect",
            mu=alpha_dbs_effect_mean_global,
            sigma=alpha_dbs_effect_sig_global,
            lower=-1,
            upper=1,
            dims="subjects",
        )

        # Calculate learning rate for each subject in each dbs condition
        alpha = pm.Deterministic(
            "alpha",
            pt.clip(
                alpha_subject.dimshuffle(0, "x")
                * (
                    1
                    + alpha_subject_dbs_effect.dimshuffle(0, "x")
                    * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
                ),
                0.01,
                1.0,
            ),
            dims=("subjects", "dbs"),
        )

        # Group-level mean and standard deviation for inverse temperature
        beta_mean_global = my_Gamma(
            "beta_mean_global",
            mu=2.5,
            sigma=1.0,
        )
        beta_sig_global = pm.Exponential("beta_sig_global", lam=5)

        # Group-level mean and standard deviation for the change of inverse temperature
        # by dbs
        beta_dbs_effect_mean_global = pm.Normal(
            "beta_dbs_effect_mean_global", mu=0.0, sigma=0.2
        )
        beta_dbs_effect_sig_global = pm.Exponential(
            "beta_dbs_effect_sig_global", lam=50
        )

        # subject-level inverse temperature
        beta_subject = my_Gamma(
            "beta_subject",
            mu=beta_mean_global,
            sigma=beta_sig_global,
            dims="subjects",
        )

        # subject-level change of inverse temperature by dbs
        beta_subject_dbs_effect = pm.TruncatedNormal(
            "beta_subject_dbs_effect",
            mu=beta_dbs_effect_mean_global,
            sigma=beta_dbs_effect_sig_global,
            lower=-1,
            upper=1,
            dims="subjects",
        )

        # Calculate inverse temperature for each subject in each dbs condition
        beta = pm.Deterministic(
            "beta",
            pt.clip(
                beta_subject.dimshuffle(0, "x")
                * (
                    1
                    + beta_subject_dbs_effect.dimshuffle(0, "x")
                    * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
                ),
                0.01,
                1000.0,
            ),
            dims=("subjects", "dbs"),
        )

        # compute the probability of selecting action 1 after each trial based on parameters
        action_is_one_probs = pm.Deterministic(
            "action_is_one_probs",
            get_probabilities_single(
                coords,
                alpha,
                beta,
                actions_arr,
                rewards_arr,
            ),
        )

        # observed data (actions are either 0 or 1) can be modeled as Bernoulli
        # likelihood with the computed probabilities
        pm.Bernoulli(
            name="like",
            p=action_is_one_probs,
            observed=observed_data,
        )

    idata_single = analyze_model(
        name="single",
        model=m_bernoulli_single,
        save_folder=save_folder,
        rng=rng,
        true_alpha_plus_2d=true_alpha_plus_2d,
        true_alpha_minus_2d=true_alpha_minus_2d,
        true_beta_2d=true_beta_2d,
        chains=4,
        tune=tune,
        draws=draws,
        draws_prior=draws_prior,
    )

    # model with two learning rates
    with pm.Model(coords=coords) as m_bernoulli_double:
        # observed data
        observed_data = pm.Data("observed_data", observed_arr)

        # Group-level mean and standard deviation for learning rate
        alpha_plus_mean_global = pm.TruncatedNormal(
            "alpha_plus_mean_global",
            mu=0.3,
            sigma=0.1,
            lower=0.1,
            upper=0.6,
        )
        alpha_plus_sig_global = pm.Exponential("alpha_plus_sig_global", lam=60)

        # Group-level mean and standard deviation for the change of learning rate by dbs
        alpha_plus_dbs_effect_mean_global = pm.Normal(
            "alpha_plus_dbs_effect_mean_global", mu=0.0, sigma=0.2
        )
        alpha_plus_dbs_effect_sig_global = pm.Exponential(
            "alpha_plus_dbs_effect_sig_global", lam=50
        )

        # subject-level learning rate
        alpha_plus_subject = pm.TruncatedNormal(
            "alpha_plus_subject",
            mu=alpha_plus_mean_global,
            sigma=alpha_plus_sig_global,
            lower=0,
            upper=1,
            dims="subjects",
        )

        # subject-level change of learning rate by dbs
        alpha_plus_subject_dbs_effect = pm.TruncatedNormal(
            "alpha_plus_subject_dbs_effect",
            mu=alpha_plus_dbs_effect_mean_global,
            sigma=alpha_plus_dbs_effect_sig_global,
            lower=-1,
            upper=1,
            dims="subjects",
        )

        # Calculate learning rate for each subject in each dbs condition
        alpha_plus = pm.Deterministic(
            "alpha_plus",
            pt.clip(
                alpha_plus_subject.dimshuffle(0, "x")
                * (
                    1
                    + alpha_plus_subject_dbs_effect.dimshuffle(0, "x")
                    * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
                ),
                0.01,
                1.0,
            ),
            dims=("subjects", "dbs"),
        )

        # Group-level mean and standard deviation for learning rate
        alpha_minus_mean_global = pm.TruncatedNormal(
            "alpha_minus_mean_global",
            mu=0.3,
            sigma=0.1,
            lower=0.1,
            upper=0.6,
        )
        alpha_minus_sig_global = pm.Exponential("alpha_minus_sig_global", lam=60)

        # Group-level mean and standard deviation for the change of learning rate by dbs
        alpha_minus_dbs_effect_mean_global = pm.Normal(
            "alpha_minus_dbs_effect_mean_global", mu=0.0, sigma=0.2
        )
        alpha_minus_dbs_effect_sig_global = pm.Exponential(
            "alpha_minus_dbs_effect_sig_global", lam=50
        )

        # subject-level learning rate
        alpha_minus_subject = pm.TruncatedNormal(
            "alpha_minus_subject",
            mu=alpha_minus_mean_global,
            sigma=alpha_minus_sig_global,
            lower=0,
            upper=1,
            dims="subjects",
        )

        # subject-level change of learning rate by dbs
        alpha_minus_subject_dbs_effect = pm.TruncatedNormal(
            "alpha_minus_subject_dbs_effect",
            mu=alpha_minus_dbs_effect_mean_global,
            sigma=alpha_minus_dbs_effect_sig_global,
            lower=-1,
            upper=1,
            dims="subjects",
        )

        # Calculate learning rate for each subject in each dbs condition
        alpha_minus = pm.Deterministic(
            "alpha_minus",
            pt.clip(
                alpha_minus_subject.dimshuffle(0, "x")
                * (
                    1
                    + alpha_minus_subject_dbs_effect.dimshuffle(0, "x")
                    * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
                ),
                0.01,
                1.0,
            ),
            dims=("subjects", "dbs"),
        )

        # Group-level mean and standard deviation for inverse temperature
        beta_mean_global = my_Gamma(
            "beta_mean_global",
            mu=2.5,
            sigma=1.0,
        )
        beta_sig_global = pm.Exponential("beta_sig_global", lam=5)

        # Group-level mean and standard deviation for the change of inverse temperature
        # by dbs
        beta_dbs_effect_mean_global = pm.Normal(
            "beta_dbs_effect_mean_global", mu=0.0, sigma=0.2
        )
        beta_dbs_effect_sig_global = pm.Exponential(
            "beta_dbs_effect_sig_global", lam=50
        )

        # subject-level inverse temperature
        beta_subject = my_Gamma(
            "beta_subject",
            mu=beta_mean_global,
            sigma=beta_sig_global,
            dims="subjects",
        )

        # subject-level change of inverse temperature by dbs
        beta_subject_dbs_effect = pm.TruncatedNormal(
            "beta_subject_dbs_effect",
            mu=beta_dbs_effect_mean_global,
            sigma=beta_dbs_effect_sig_global,
            lower=-1,
            upper=1,
            dims="subjects",
        )

        # Calculate inverse temperature for each subject in each dbs condition
        beta = pm.Deterministic(
            "beta",
            pt.clip(
                beta_subject.dimshuffle(0, "x")
                * (
                    1
                    + beta_subject_dbs_effect.dimshuffle(0, "x")
                    * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
                ),
                0.01,
                1000.0,
            ),
            dims=("subjects", "dbs"),
        )

        # compute the probability of selecting action 1 after each trial based on parameters
        action_is_one_probs = pm.Deterministic(
            "action_is_one_probs",
            get_probabilities_double(
                coords,
                alpha_plus,
                alpha_minus,
                beta,
                actions_arr,
                rewards_arr,
            ),
        )

        # observed data (actions are either 0 or 1) can be modeled as Bernoulli
        # likelihood with the computed probabilities
        pm.Bernoulli(
            name="like",
            p=action_is_one_probs,
            observed=observed_data,
        )

    idata_double = analyze_model(
        name="double",
        model=m_bernoulli_double,
        save_folder=save_folder,
        rng=rng,
        true_alpha_plus_2d=true_alpha_plus_2d,
        true_alpha_minus_2d=true_alpha_minus_2d,
        true_beta_2d=true_beta_2d,
        chains=4,
        tune=tune,
        draws=draws,
        draws_prior=draws_prior,
    )

    # model comparison using LOO (Leave-One-Out cross-validation)
    df_comp_loo = az.compare(
        {"m_bernoulli_single": idata_single, "m_bernoulli_double": idata_double}
    )
    # print the comparison table in a text file
    with open(f"{save_folder}/model_comparison.txt", "w") as f:
        f.write(str(df_comp_loo))

    # plot results of the model comparison
    az.plot_compare(df_comp_loo, insample_dev=False)
    plt.savefig(f"{save_folder}/model_comparison.png")
