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
import pandas as pd
import pickle
from scipy.special import logsumexp
from scipy.optimize import minimize
import seaborn as sns
import sys


def generate_data_q_learn(rng, alpha_plus, alpha_minus, beta, n=100, p=0.2):
    """
    This function generates data from a simple Q-learning model. It simulates a
    two-armed bandit task with actions 0 and 1, and reward probabilities P(R(a=1)) = p
    and P(R(a=0)) = 1-p. After half of the trials the propapilities reverse.The
    Q-learning model uses softmax action selection and two learning rates, alpha_plus
    and alpha_minus, for positive and negative prediction errors, respectively.

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
            Probability of reward for the second action (action==1).

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
        # reverse the reward propapilities
        if i == n // 2:
            prob_r.reverse()
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

    # remove max for numerical stability
    Qs = Qs - pt.max(Qs, axis=1, keepdims=True)

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

    # remove max for numerical stability
    Qs = Qs - pt.max(Qs, axis=1, keepdims=True)

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
    chains,
    tune,
    draws,
    draws_prior,
    target_accept,
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
        idata.extend(
            pm.sample(
                random_seed=rng,
                chains=chains,
                tune=tune,
                draws=draws,
                target_accept=target_accept,
            )
        )
        # compute the log likelihood of the model
        pm.compute_log_likelihood(idata)

    # plot the posterior distributions
    for var_name in idata.posterior.data_vars:
        # skip deterministic variables
        if var_name == "action_is_one_probs":
            continue
        az.plot_posterior(
            data=idata,
            var_names=[var_name],
        )
        plt.savefig(f"{save_folder}/{name}_posterior_{var_name}.png")
        plt.close("all")

    # plot the forest plot
    if (
        ("alpha" in idata.posterior.data_vars) and ("beta" in idata.posterior.data_vars)
    ) or (
        ("alpha_plus" in idata.posterior.data_vars)
        and ("alpha_minus" in idata.posterior.data_vars)
        and ("beta" in idata.posterior.data_vars)
    ):
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


def load_data_previously_selected(
    subject_type: str,
    shortcut_type: str,
    dbs_state: str,
    dbs_variant: str,
):
    """
    For all subjects load the choices and the obtained rewards for all trials.

    Args:
        subject_type (str):
            "patient" or "simulation"
        shortcut_type (str):
            "plastic" or "fixed"
        dbs_state (str):
            "ON" or "OFF"
        dbs_variant (str):
            "off", "suppression", "efferent", "afferent", "passing", or "dbs-all"

    Returns:
        pd.DataFrame:
            DataFrame containing the subject, trial, choice and reward columns
    """
    # data needs to be loaded differently for patients/simulations
    if subject_type == "patient":
        file_name = "data/patient_data/choices_rewards_per_trial.pkl"
        # load data using pickle
        with open(file_name, "rb") as f:
            data_patients = pickle.load(f)
        # get the correct format for the data
        ret = {}
        ret["subject"] = []
        ret["trial"] = []
        ret["choice"] = []
        ret["reward"] = []
        for subject in data_patients:
            for trial, choice in enumerate(
                data_patients[subject][dbs_state]["choices"]
            ):
                ret["subject"].append(subject)
                ret["trial"].append(trial)
                ret["choice"].append(choice)
            for reward in data_patients[subject][dbs_state]["rewards"]:
                ret["reward"].append(reward)
        # choices should be 0 and 1
        ret["choice"] = (
            transform_range(np.array(ret["choice"]), new_min=0, new_max=1)
            .astype(int)
            .tolist()
        )

    elif subject_type == "simulation":
        shortcut_load = {"plastic": 1, "fixed": 0}[shortcut_type]
        if dbs_state == "OFF":
            dbs_load = 0
        elif dbs_state == "ON":
            dbs_load = {
                "off": 0,
                "suppression": 1,
                "efferent": 2,
                "afferent": 3,
                "passing": 4,
                "dbs-all": 5,
            }[dbs_variant]
        file_name = (
            lambda sim_id: f"data/simulation_data/choices_rewards_per_trial_Shortcut{shortcut_load}_DBS_State{dbs_load}_sim{sim_id}.pkl"
        )
        # load data using pickle
        # get the correct format for the data
        ret = {}
        ret["subject"] = []
        ret["trial"] = []
        ret["choice"] = []
        ret["reward"] = []
        for sim_id in range(100):
            with open(file_name(sim_id), "rb") as f:
                data_patients = pickle.load(f)
                for trial, choice in enumerate(data_patients["choices"]):
                    ret["subject"].append(sim_id)
                    ret["trial"].append(trial)
                    # choice+1 to obtain choices 1 and 2 as for patients
                    ret["choice"].append(choice + 1)
                for reward in data_patients["rewards"]:
                    ret["reward"].append(reward)

    return pd.DataFrame(ret)


def transform_range(vector: np.ndarray, new_min, new_max):
    return ((vector - vector.min()) / (vector.max() - vector.min())) * (
        new_max - new_min
    ) + new_min


def llik_td(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    alpha, beta = x
    actions, rewards = args

    # Initialize values
    Qs = np.zeros((len(actions), 2))
    Q = np.array([0.5, 0.5])
    logp_actions = np.zeros(len(actions))

    for t, (a, r) in enumerate(zip(actions, rewards)):
        Qs[t] = Q
        # Apply the softmax transformation
        Q_ = Q * beta
        logp_action = Q_ - logsumexp(Q_)

        # Store the log probability of the observed action
        logp_actions[t] = logp_action[a]

        # Update the Q values for the next trial
        Q[a] = Q[a] + alpha * (r - Q[a])

    # Return the negative log likelihood of all observed actions
    return -np.sum(logp_actions[1:]), Qs


def mean_without_outlier(data: np.ndarray):
    # Calculate Q1 and Q3 (25th and 75th percentiles)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1  # Interquartile range

    # Set bounds to exclude outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter data to remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    mean_without_outliers = np.mean(filtered_data)
    std_without_outliers = np.std(filtered_data)

    return mean_without_outliers, std_without_outliers


def get_mle_estimates(
    data_on, data_off, plot_patients, rng, plot_mle_estimates, save_folder
):
    alpha_patients_arr = np.empty((len(data_on["subject"].unique()), 2))
    beta_patients_arr = np.empty((len(data_on["subject"].unique()), 2))

    for dbs, data in enumerate([data_off, data_on]):
        for subject_idx, subject in enumerate(data["subject"].unique()):
            actions = data[data["subject"] == subject]["choice"].values
            rewards = data[data["subject"] == subject]["reward"].values

            result = minimize(
                lambda x, *args: llik_td(x, *args)[0],
                [0.3, 1.0],
                args=(
                    actions,
                    rewards,
                ),
                method="BFGS",
            )
            alpha, beta = result.x
            alpha_patients_arr[subject_idx, dbs] = alpha
            beta_patients_arr[subject_idx, dbs] = beta
            if plot_patients:
                qs = llik_td(
                    result.x,
                    *(
                        actions,
                        rewards,
                    ),
                )[1]
                fake_actions, fake_rewards, fake_qs = generate_data_q_learn(
                    rng,
                    alpha,
                    alpha,
                    beta,
                    len(actions),
                )
                # plot the qs and which action selected and rewards
                actions = transform_range(actions, new_min=-1, new_max=1).astype(int)
                fake_actions = transform_range(
                    fake_actions, new_min=-1, new_max=1
                ).astype(int)
                plt.figure(figsize=(6.4 * 2, 4.8 * 3))
                plt.subplot(321)
                plt.title(f"DBS {['OFF', 'ON'][dbs]} Subject {subject}")
                plt.bar(
                    range(len(actions)),
                    actions * (actions > 0).astype(int),
                    width=1.0,
                    color="b",
                )
                plt.bar(
                    range(len(actions)),
                    actions * (actions < 0).astype(int),
                    width=1.0,
                    color="r",
                )
                plt.subplot(323)
                plt.bar(range(len(actions)), rewards, width=1.0)
                plt.subplot(325)
                plt.title(f"alpha {round(alpha, 3)}, beta {round(beta, 3)}")
                plt.plot(range(len(actions)), qs[:, 0], color="r")
                plt.plot(range(len(actions)), qs[:, 1], color="b")
                plt.ylim(0.5 - np.abs(qs - 0.5).max(), 0.5 + np.abs(qs - 0.5).max())

                plt.subplot(322)
                plt.title("fake data")
                plt.bar(
                    range(len(actions)),
                    fake_actions * (fake_actions > 0).astype(int),
                    width=1.0,
                    color="b",
                )
                plt.bar(
                    range(len(actions)),
                    fake_actions * (fake_actions < 0).astype(int),
                    width=1.0,
                    color="r",
                )
                plt.subplot(324)
                plt.bar(range(len(actions)), fake_rewards, width=1.0)
                plt.subplot(326)
                plt.title(f"alpha {round(alpha, 3)}, beta {round(beta, 3)}")
                plt.plot(range(len(actions)), fake_qs[:, 0], color="r")
                plt.plot(range(len(actions)), fake_qs[:, 1], color="b")
                plt.ylim(
                    0.5 - np.abs(fake_qs - 0.5).max(), 0.5 + np.abs(fake_qs - 0.5).max()
                )
                plt.tight_layout()
                plt.savefig(f"{save_folder}/data_patient_{subject}_dbs_{dbs}.png")
                plt.close("all")
    if plot_mle_estimates:
        # plot MLE alphas and betas of patients

        # Combine the data into a single list for boxplot
        combined_data = [
            alpha_patients_arr[:, 0],
            alpha_patients_arr[:, 1],
            beta_patients_arr[:, 0],
            beta_patients_arr[:, 1],
        ]

        # Create a list of labels for each boxplot
        labels = [
            "Alpha - DBS OFF",
            "Alpha - DBS ON",
            "Beta - DBS OFF",
            "Beta - DBS ON",
        ]

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=combined_data)

        # Set x-axis labels
        plt.ylim(0, 15)
        plt.xticks(ticks=np.arange(4), labels=labels, rotation=45)
        plt.ylabel("Value")
        plt.title("Boxplot Comparison Between Two Conditions")
        plt.tight_layout()
        plt.savefig(f"{save_folder}/data_mle_estimates_boxplot.png")
        plt.close("all")

    # get mean and std estimates for alpha and beta for the two groups of subjects
    alpha_mle_estimates = mean_without_outlier(alpha_patients_arr.flatten())
    beta_mle_estimates = mean_without_outlier(beta_patients_arr.flatten())

    # write mle estimates of alpha and beta to a file and their estimates in logit and
    # log space
    alpha_m_logit, alpha_s_logit = prior_in_logit(
        alpha_mle_estimates[0], alpha_mle_estimates[1]
    )
    beta_m_log, beta_s_log = prior_in_log(beta_mle_estimates[0], beta_mle_estimates[1])
    with open(f"{save_folder}/mle_estimates.txt", "w") as f:
        f.write(f"alpha: {alpha_mle_estimates[0]} +/- {alpha_mle_estimates[1]}\n")
        f.write(f"alpha: {alpha_m_logit} +/- {alpha_s_logit} in logit space\n")
        f.write(f"beta: {beta_mle_estimates[0]} +/- {beta_mle_estimates[1]}\n")
        f.write(f"beta: {beta_m_log} +/- {beta_s_log} in log space\n")

    return alpha_mle_estimates, beta_mle_estimates


def prior_in_logit(mu, sigma):
    """
    Transform a prior (mean and standard deviation) from the "probability space" x=(0,1)
    to the logit space x=(-inf,inf) y=(0,1).

    !!! warning
    Works worse if mu - sigma > 0 and mu + sigma < 1.

    Args:
        mu (float):
            Mean of the prior in the probability space (between 0 and 1).
        sigma (float):
            Standard deviation of the prior in the probability space.

    Returns:
        mu_logit (float):
            Mean of the prior in the logit space.
        sigma_logit (float):
            Standard deviation of the prior in the logit space.
    """
    mu_logit = np.log(mu / (1 - mu))
    lower = np.clip((mu - sigma), 0.0001, 0.9999)
    upper = np.clip((mu + sigma), 0.0001, 0.9999)
    sigma_interval_lower = np.log(lower / (1 - lower))
    sigma_interval_upper = np.log(upper / (1 - upper))
    sigma_logit = (sigma_interval_upper - sigma_interval_lower) / 2

    return mu_logit, sigma_logit


def prior_in_log(mu, sigma):
    """
    Transform a prior (mean and standard deviation) from the "rate space" x=(0,inf)
    to the log space x=(-inf,inf) y=(0,inf).


    Args:
        mu (float):
            Mean of the prior in the probability space (>0).
        sigma (float):
            Standard deviation of the prior in the probability space.

    Returns:
        mu_log (float):
            Mean of the prior in the log space.
        sigma_log (float):
            Standard deviation of the prior in the log space.
    """
    mu_log = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
    sigma_log = np.sqrt(np.log(1 + sigma**2 / mu**2))

    return mu_log, sigma_log


def load_inferred_params_and_create_data():

    # load the inference data object
    save_folder = "results_fitting_q_learning_with_dbs_effects"
    idata = az.from_netcdf(f"{save_folder}/single_idata.nc")

    # get the traces for parameters `alpha` and `beta`
    alpha = idata.posterior["alpha"].values
    beta = idata.posterior["beta"].values

    # combine chains
    alpha = alpha.reshape((np.prod(alpha.shape[:2]),) + alpha.shape[2:])
    beta = beta.reshape((np.prod(beta.shape[:2]),) + beta.shape[2:])

    alpha = alpha[:3]
    beta = beta[:3]

    rng = np.random.default_rng(12345)

    for idx, alpha_beta in enumerate(zip(alpha, beta)):
        alpha, beta = alpha_beta
        for subject in range(2):
            for dbs in range(2):
                fake_actions, fake_rewards, fake_qs = generate_data_q_learn(
                    rng,
                    alpha[subject, dbs],
                    alpha[subject, dbs],
                    beta[subject, dbs],
                    120,
                )
                # plot the qs and which action selected and rewards
                fake_actions = transform_range(
                    fake_actions, new_min=-1, new_max=1
                ).astype(int)
                plt.figure(figsize=(6.4 * 2, 4.8 * 3))
                plt.subplot(321)
                plt.subplot(323)
                plt.subplot(325)

                plt.subplot(322)
                plt.title("fake data")
                plt.bar(
                    range(120),
                    fake_actions * (fake_actions > 0).astype(int),
                    width=1.0,
                    color="b",
                )
                plt.bar(
                    range(120),
                    fake_actions * (fake_actions < 0).astype(int),
                    width=1.0,
                    color="r",
                )
                plt.subplot(324)
                plt.bar(range(120), fake_rewards, width=1.0)
                plt.subplot(326)
                plt.title(
                    f"alpha {round(alpha[subject, dbs], 3)}, beta {round(beta[subject, dbs], 3)}"
                )
                plt.plot(range(120), fake_qs[:, 0], color="r")
                plt.plot(range(120), fake_qs[:, 1], color="b")
                plt.ylim(
                    0.5 - np.abs(fake_qs - 0.5).max(), 0.5 + np.abs(fake_qs - 0.5).max()
                )
                plt.tight_layout()
                plt.savefig(
                    f"{save_folder}/data_patient_{subject}_dbs_{dbs}_fake{idx}.png"
                )
                plt.close("all")


def create_param(
    param_name: str,
    coords: dict,
    estimates: tuple,
    std_of_mean: float,
    dbs_effect: tuple,
    max: float,
):
    """
    Create a parameter in the model with a hierarchical structure. The parameter priors
    are defined in a transformed logit space (group level mean, standard deviation of the
    group level mean, standard deviation of the group of subjects). The parameter is
    transformed to the original space using a sigmoid function times max.

    Args:
        param_name (str):
            Name of the parameter.
        coords (dict):
            Dictionary containing the coordinates of the model.
        estimates (tuple):
            Tuple containing the group mean and standard deviation of the group of
            subjects (in the original / the parameter's space).
        std_of_mean (float):
            Standard deviation of the group-level mean (in the original / the
            parameter's space).
        dbs_effect (tuple):
            Tuple containing the mean and standard deviation of the change of the
            parameter by dbs (in the original / the parameter's space).
        max (float):
            Maximum value of the parameter in the original (the parameter's) space.

    Returns:
        pm.Deterministic:
            Deterministic variable representing the parameter in the original space.
    """
    # Group-level mean and standard deviation for parameter in the transformed space
    mu_transformed, sigma_transformed = prior_in_logit(
        estimates[0] / max, estimates[1] / max
    )
    _, std_of_mean_transformed = prior_in_logit(estimates[0] / max, std_of_mean / max)
    mean_global = pm.Normal(
        f"{param_name}_mean_global", mu_transformed, std_of_mean_transformed
    )
    sig_global = pm.Exponential(f"{param_name}_sig_global", 1.0 / sigma_transformed)
    # subject-level parameter non-centered parameterization
    z_subject = pm.Normal(f"{param_name}_z_subject", 0, 1, dims="subjects")

    # Group-level mean and standard deviation for the change of parameter by dbs in the
    # transformed space
    _, dbs_effect_0_transformed = prior_in_logit(
        estimates[0] / max, dbs_effect[0] / max
    )
    _, dbs_effect_1_transformed = prior_in_logit(
        estimates[0] / max, dbs_effect[1] / max
    )
    dbs_effect_mean_global = pm.Normal(
        f"{param_name}_dbs_effect_mean_global", mu=0.0, sigma=dbs_effect_0_transformed
    )
    dbs_effect_sig_global = pm.Exponential(
        f"{param_name}_dbs_effect_sig_global", lam=1.0 / dbs_effect_1_transformed
    )
    # subject-level change of parameter by dbs non-centered parameterization
    z_subject_dbs_effect = pm.Normal(
        f"{param_name}_z_subject_dbs_effect", 0, 1, dims="subjects"
    )

    # Calculate parameter for each subject in each dbs condition in the transformed space
    # and then transform to the original space
    return pm.Deterministic(
        param_name,
        max
        * pm.math.sigmoid(
            (mean_global + z_subject * sig_global).dimshuffle(0, "x")
            + (
                (
                    dbs_effect_mean_global
                    + z_subject_dbs_effect * dbs_effect_sig_global
                ).dimshuffle(0, "x")
                * pt.arange(len(coords["dbs"])).dimshuffle("x", 0)
            )
        ),
        dims=("subjects", "dbs"),
    )


def estimate_p_explore_of_patients(
    data_on, data_off, plot_patients, rng, plot_mle_estimates, save_folder
):
    # load the inference data object
    save_folder = "results_fitting_q_learning_complete"
    idata = az.from_netcdf(f"{save_folder}/double_idata.nc")
    number_samples = np.prod(idata.posterior["beta"].values.shape[:2])

    posterior = {}
    for parameter in ["alpha_plus", "alpha_minus", "beta"]:
        vals = idata.posterior[parameter].values
        # first two dimensions are the chains and samples per chain -> combine them
        posterior[parameter] = vals.reshape((number_samples,) + vals.shape[2:])

    alpha_patients_arr = np.empty((len(data_on["subject"].unique()), 2))
    beta_patients_arr = np.empty((len(data_on["subject"].unique()), 2))
    # loop over patients data
    for dbs, data in enumerate([data_off, data_on]):
        for subject_idx, subject in enumerate(data["subject"].unique()):
            actions = data[data["subject"] == subject]["choice"].values
            rewards = data[data["subject"] == subject]["reward"].values


if __name__ == "__main__":

    save_folder = f"results_fitting_q_learning_complete_new/{sys.argv[1]}"
    seed = 123
    tune = 7000
    draws = 15000
    tune = 500  # TODO remove
    draws = 1000  # TODO remove
    draws_prior = 2000
    target_accept = 0.975
    plot_patients = True
    plot_mle_estimates = True
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    az.style.use("arviz-darkgrid")
    rng = np.random.default_rng(seed)

    if sys.argv[1] != "comparison":

        # loading patient data
        data_off = load_data_previously_selected(
            subject_type="patient",
            shortcut_type=None,
            dbs_state="OFF",
            dbs_variant=None,
        )
        data_on = load_data_previously_selected(
            subject_type="patient",
            shortcut_type=None,
            dbs_state="ON",
            dbs_variant=None,
        )
        n_subjects = len(data_off["subject"].unique())
        n_subjects = 2  # TODO remove

        # get mle estimates of alpha and beta for the patient data, patient data already
        # only contains patients which did both dbs on and off
        alpha_estimates, beta_estimates = get_mle_estimates(
            data_on=data_on,
            data_off=data_off,
            plot_patients=plot_patients,
            rng=rng,
            plot_mle_estimates=plot_mle_estimates,
            save_folder=save_folder,
        )

        # get actions, rewards, observed data arrays
        actions_arr = np.empty((2, n_subjects), dtype=object)
        rewards_arr = np.empty((2, n_subjects), dtype=object)
        observed_list = []

        for dbs in [0, 1]:
            data = data_off if dbs == 0 else data_on
            for subject_idx, subject in enumerate(data_off["subject"].unique()):
                if subject_idx >= n_subjects:
                    break
                actions = data[data["subject"] == subject]["choice"].values
                actions_arr[dbs, subject_idx] = actions
                rewards_arr[dbs, subject_idx] = data[data["subject"] == subject][
                    "reward"
                ].values
                observed_list.append(actions[1:])

        observed_arr = np.concatenate(observed_list)

        coords = {
            "dbs": range(2),
            "subjects": range(n_subjects),
        }

    if sys.argv[1] == "single":
        # model with single learning rate
        with pm.Model(coords=coords) as m_bernoulli_single:
            # observed data
            observed_data = pm.Data("observed_data", observed_arr)

            alpha = create_param(
                param_name="alpha",
                coords=coords,
                estimates=(alpha_estimates[0], 0.1),
                std_of_mean=0.1,
                dbs_effect=(0.1, 0.1),
                max=1.0,
            )

            beta = create_param(
                param_name="beta",
                coords=coords,
                estimates=(beta_estimates[0], 2.0),
                std_of_mean=1.0,
                dbs_effect=(3.0, 1.0),
                max=20.0,
            )

            # compute the probability of selecting action 1 after each trial, based on
            # parameters
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
            chains=4,
            tune=tune,
            draws=draws,
            draws_prior=draws_prior,
            target_accept=target_accept,
        )

    elif sys.argv[1] == "double":
        # model with two learning rates
        with pm.Model(coords=coords) as m_bernoulli_double:
            # observed data
            observed_data = pm.Data("observed_data", observed_arr)

            alpha_plus = create_param(
                param_name="alpha_plus",
                coords=coords,
                estimates=(alpha_estimates[0], 0.1),
                std_of_mean=0.1,
                dbs_effect=(0.1, 0.1),
                max=1.0,
            )

            alpha_minus = create_param(
                param_name="alpha_minus",
                coords=coords,
                estimates=(alpha_estimates[0], 0.1),
                std_of_mean=0.1,
                dbs_effect=(0.1, 0.1),
                max=1.0,
            )

            beta = create_param(
                param_name="beta",
                coords=coords,
                estimates=(beta_estimates[0], 2.0),
                std_of_mean=1.0,
                dbs_effect=(3.0, 1.0),
                max=20.0,
            )

            # compute the probability of selecting action 1 after each trial based on
            # parameters
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
            chains=4,
            tune=tune,
            draws=draws,
            draws_prior=draws_prior,
            target_accept=target_accept,
        )

    elif sys.argv[1] == "comparison":

        # load the inference data objects
        idata_single = az.from_netcdf(
            f"{save_folder[:-len(sys.argv[1])] + '/single'}single_idata.nc"
        )
        idata_double = az.from_netcdf(
            f"{save_folder[:-len(sys.argv[1])] + '/double'}double_idata.nc"
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
