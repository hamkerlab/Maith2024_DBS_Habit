import ANNarchy as ann
import BG_model as BG
import numpy as np
import pandas as pd
import sys
import random
from CompNeuroPy import save_variables
import pickle


############################## fixed random values ##################################

seed = int(sys.argv[1]) + 17112023  # 17112023 = Date
random.seed(seed)
ann_compile_str = f"annarchy_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}"


############################### Reset #################################
def reset_activity():
    "Resets activity in the network by setting all inputs to 0.0"
    BG.IT.B = 0.0
    BG.PPN.B = 0.0
    BG.SNc.alpha = 0.0
    ann.simulate(200.0)


def weighted_mean(arr):
    """
    Apply a weighted mean to the array. The weights are the inverse indices of the
    array so that the first element has the highest weight and the last element the
    lowest weight.
    """
    return np.average(arr, weights=np.arange(len(arr))[::-1])


def get_support(monitor_support: ann.Monitor):
    """
    From the thalamic inputs recorded with the monitor, calculate the support for the
    action 0 neuron from the inhibitory and excitatory inputs.

    Returns:
    support_exc: The support from the excitatory inputs
    support_inh: The support from the inhibitory inputs
    """
    # get sum(exc) and sum(inh) from monitor, first index is the time, second index
    # is the neuron index
    sum_exc = monitor_support.get("sum(exc)")
    sum_inh = monitor_support.get("sum(inh)")

    # from sum_exc calculate support_exc by substracting the values of neuron 1 from
    # neuron 0
    support_exc = sum_exc[:, 0] - sum_exc[:, 1]
    # average over time while weighting earlier time points higher
    support_exc_ret = weighted_mean(support_exc)

    # from sum_inh calculate support_inh by substracting the values of neuron 0 from
    # neuron 1
    support_inh = sum_inh[:, 1] - sum_inh[:, 0]
    # average over time while weighting earlier time points higher
    support_inh_ret = weighted_mean(support_inh)

    # pause the monitor
    monitor_support.pause()

    return support_exc_ret, support_inh_ret


##############################################################################
################################# test run ###################################
##############################################################################


def trial(
    k,
    immer,
    nie,
    belohnung,
    umkehr,
    anz_trials,
    reward,
    IT,
    PPN,
    SNc,
    PFC,
    dbs_state,
    DBS,
    GPi,
    monitor_support: ann.Monitor,
):
    reward_setzen = False
    idx1 = 1
    idx2 = 2

    # start value
    IT.B = 0.0
    ann.simulate(100.0)

    # set input
    IT.B = 1.0

    # start monitor recording sum(exc) and sum(inh)
    monitor_support.resume()

    # simulate max 3 seconds or when the threshhold is reached
    PFC.stop_condition = "(r > 0.8)"
    ann.simulate_until(max_duration=3000.0, population=PFC)

    ######################## reversal task on/off ##############################
    if umkehr == True:
        if k < anz_trials / 2:
            idx1 = 0
            idx2 = 1
        else:
            idx1 = 1
            idx2 = 0

    ############################## set reward ##################################
    # np.argmax(PFC.r) -> index with biggest fireing rate
    selected = np.argmax(PFC.r)

    # get support from monitor support and stop monitor
    s_exc, s_inh = get_support(monitor_support, selected)

    # always reward
    if immer:
        if (selected == 0) or (selected == 1):
            reward_setzen = True

    # never reward
    if nie:
        if (selected == 0) or (selected == 1):
            reward_setzen = False

    # random reward
    if immer == False and nie == False:
        if (selected == idx1 and reward[k] == 1) or (
            selected == idx2 and reward[k] == 0
        ):
            reward_setzen = True

    ############################### set reward ##################################
    if reward_setzen:
        PPN.B = 1.0
        ann.simulate(1)
        SNc.alpha = 1.0
        ann.simulate(60)
        erfolg = 1.0
    else:
        PPN.B = 0.0
        ann.simulate(1)
        SNc.alpha = 1.0
        ann.simulate(60)
        erfolg = 0.0

    SNc.alpha = 0.0
    PPN.B = 0.0

    ann.simulate(100)

    return erfolg, selected, s_exc, s_inh


##############################################################################
############################ reward deitribution #############################
##############################################################################
def create_reward(anz_trials, p, zufall):
    reward = []

    if zufall:
        for i in range(int(p / 100 * anz_trials)):
            reward.append(1)
        for i in range(int((100 - p) / 100 * anz_trials)):
            reward.append(0)
        random.shuffle(reward)
        return reward
    elif p == 100:
        reward = np.ones(anz_trials, int)
        return reward
    else:
        w = int(anz_trials * ((100 - p) / 100))
        start = int((anz_trials / w) - 1)
        step = int(anz_trials / w)

        reward = np.ones(anz_trials, int)
        for i in range(start, anz_trials, step):
            reward[i] = int(0)
        return reward


###############################################################################
################################# save data ###################################
###############################################################################

####################### save rewards per session ##############################


def save_data(
    success, selected_list, support_list, dbs_state, shortcut, plastic_weights
):
    success = [int(x) for x in success]

    column = int(sys.argv[1])
    df = pd.DataFrame(success)

    save_data = sys.argv[6]
    if save_data == "True":
        # filename
        filepath = f"data/simulation_data/Results_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}.json"
        df.to_json(filepath, orient="records", lines=True)

        # save success and selected_list separately using pickle
        with open(
            f"data/simulation_data/choices_rewards_per_trial_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}.pkl",
            "wb",
        ) as f:
            pickle.dump({"rewards": success, "choices": selected_list}, f)

        # save plastic weights
        save_variables(
            variable_list=[plastic_weights],
            name_list=[
                f"plastic_weights_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}"
            ],
            path="data/simulation_data/",
        )

        # save support values
        support_dict = {
            "support_exc": np.array(support_list)[:, 0],
            "support_inh": np.array(support_list)[:, 1],
        }
        save_variables(
            variable_list=[support_dict],
            name_list=[
                f"support_values_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}"
            ],
            path="data/simulation_data/",
        )


###############################################################################
############################## save parameter data ############################
###############################################################################

############################ save rewards per session #########################


def save_parameter(success, dbs_state, shortcut, parameter):
    success = [int(x) for x in success]
    step = int(sys.argv[9])
    column = int(sys.argv[1])

    ################################ save data ################################

    if dbs_state == 1:
        filepath = f"data/parameter_data/1_suppression/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 2:
        filepath = f"data/parameter_data/2_efferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 3:
        filepath = f"data/parameter_data/3_afferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 4:
        filepath = f"data/parameter_data/4_passing_fibres/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"

    df = pd.DataFrame(success)
    df.to_json(filepath, orient="records", lines=True)

    ##################### save optimal param #####################

    if column == 0:
        if dbs_state == 1:
            filepath = f"data/parameter_data/1_suppression/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 2:
            filepath = f"data/parameter_data/2_efferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 3:
            filepath = f"data/parameter_data/3_afferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 4:
            filepath = f"data/parameter_data/4_passing_fibres/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"

        optimal_data = [parameter]
        df = pd.DataFrame(optimal_data)
        df.to_json(filepath, orient="records", lines=True)


###############################################################################
########################### save data gpi scatter #############################
###############################################################################

####################### save rewards per session ##############################


def save_GPi_r(mean_GPi, dbs_state, shortcut, parameter):
    step = int(sys.argv[9])
    column = int(sys.argv[1])

    ####################### save average rate GPi #############################

    if dbs_state == 1:
        filepath = f"data/gpi_scatter_data/1_suppression/mean_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 2:
        filepath = f"data/gpi_scatter_data/2_efferent/mean_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 3:
        filepath = f"data/gpi_scatter_data/3_afferent/mean_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
    if dbs_state == 4:
        filepath = f"data/gpi_scatter_data/4_passing_fibres/mean_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"

    mean_GPi = [mean_GPi]
    df = pd.DataFrame(mean_GPi)
    df.to_json(filepath, orient="records", lines=True)

    ############################ save gpi scatter data #################################

    if column == 0:
        if dbs_state == 1:
            filepath = f"data/gpi_scatter_data/1_suppression/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 2:
            filepath = f"data/gpi_scatter_data/2_efferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 3:
            filepath = f"data/gpi_scatter_data/3_afferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
        if dbs_state == 4:
            filepath = f"data/gpi_scatter_data/4_passing_fibres/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"

        optimal_data = [parameter]
        df = pd.DataFrame(optimal_data)
        df.to_json(filepath, orient="records", lines=True)


##############################################################################
################################ run simulation ##############################
##############################################################################


def simulate():
    dbs_state = int(sys.argv[2])
    shortcut = int(sys.argv[3])
    parameter = float(sys.argv[7])
    save_parameter_data = sys.argv[8]
    save_mean_GPi = sys.argv[10]
    dbs_param_state = int(sys.argv[11])

    ######################### compile BG_Modell ##############################

    populations = BG.create_network(
        seed,
        dbs_state,
        shortcut,
        parameter,
        dbs_param_state,
        ann_compile_str=ann_compile_str,
    )

    ####################### get population parameters ########################

    IT = populations[0]
    PFC = populations[1]
    StrD1 = populations[2]
    StrD2 = populations[3]
    STN = populations[4]
    GPe = populations[5]
    GPi = populations[6]
    Thal = populations[7]
    SNc = populations[8]
    StrThal = populations[9]
    PPN = populations[10]
    PFC = populations[11]
    ITStrD1 = populations[12]
    ITStrD2 = populations[13]
    ITSTN = populations[14]
    StrD1GPi = populations[15]
    StrD2GPe = populations[16]
    STNGPi = populations[17]
    StrD1SNc = populations[18]
    ITThal = populations[19]
    ThalPFC = populations[20]
    DBS = populations[21]

    ########################### simulation settings ############################

    # Parameter
    # always reward
    immer = False
    # never reward
    nie = False
    # reward distribution 100:0(100), 80:20(80), 70:30(70)...
    belohnungsverteilung = 80
    # random reward distribution(True) or fixed(False)
    zufall = True
    # reversal task after half of the trials ON(True) or OFF(False)
    umkehr = True
    # number of repetitions
    anz_trials = 120

    success = []
    selected_list = []
    support_list = []
    StrD1_GPi_list = []
    step = np.linspace(0, anz_trials - 1, 5).astype(int)
    reward = create_reward(anz_trials, belohnungsverteilung, zufall)
    plastic_weights = {}

    ############################# initial monitor ###############################

    monitor = BG.BGMonitor([IT, StrD1, StrD2, STN, GPe, GPi, SNc, Thal, PFC, StrThal])
    monitor_support = ann.Monitor(Thal, ["sum(exc)", "sum(inh)"])
    monitor_support.pause()

    ################## simulate 120 trials (1 simulation) ######################

    # start DBS
    if dbs_state > 0:
        DBS.on()

    # run simulation
    for t in range(anz_trials):  # number of Trials
        if t == 0:
            monitor.start()  # record firing rates
            gewichte = BG.extract_data(
                SNc,
                ITStrD1,
                ITStrD2,
                ITSTN,
                StrD1GPi,
                StrD2GPe,
                STNGPi,
                StrD1SNc,
                ITThal,
                ThalPFC,
                shortcut,
            )  # record weigths
            schwellen = BG.extract_mean(
                ITStrD1, ITStrD2, ITSTN, StrD1GPi, StrD2GPe, STNGPi
            )  # record means
            rewarded, selected, s_exc, s_inh = trial(
                t,
                immer,
                nie,
                belohnungsverteilung,
                umkehr,
                anz_trials,
                reward,
                IT,
                PPN,
                SNc,
                PFC,
                dbs_state,
                DBS,
                GPi,
                monitor_support,
            )
            success.append(rewarded)  # record rewards
            selected_list.append(selected)  # record selection
            support_list.append([s_exc, s_inh])  # record support
        else:
            rewarded, selected, s_exc, s_inh = trial(
                t,
                immer,
                nie,
                belohnungsverteilung,
                umkehr,
                anz_trials,
                reward,
                IT,
                PPN,
                SNc,
                PFC,
                dbs_state,
                DBS,
                GPi,
                monitor_support,
            )
            success.append(rewarded)  # record rewards
            selected_list.append(selected)  # record selection
            support_list.append([s_exc, s_inh])  # record support
            weigths = BG.extract_data(
                SNc,
                ITStrD1,
                ITStrD2,
                ITSTN,
                StrD1GPi,
                StrD2GPe,
                STNGPi,
                StrD1SNc,
                ITThal,
                ThalPFC,
                shortcut,
            )
            tresh = BG.extract_mean(ITStrD1, ITStrD2, ITSTN, StrD1GPi, StrD2GPe, STNGPi)

            # combine weights of trials
            for i in gewichte:
                for k in range(len(gewichte[i])):
                    gewichte[i][k] = np.append(gewichte[i][k], weigths[i][k])

            # combine means of trials
            for i in schwellen:
                for k in range(len(schwellen[i])):
                    schwellen[i][k] = np.append(schwellen[i][k], tresh[i][k])

        if t in step:
            StrD1_GPi_list.append(ITThal.w)

        if t == 119:
            recordings = monitor.get()
            monitor.stop()

        # at the end of each trial get the plastic weights
        plastic_weights = BG.get_plastic_weights(plastic_weights)

    save_data(
        success, selected_list, support_list, dbs_state, shortcut, plastic_weights
    )

    if save_parameter_data == "True" and dbs_state > 0 and dbs_state < 5:
        save_parameter(success, dbs_state, shortcut, parameter)

    if save_mean_GPi == "True" and dbs_state > 0 and dbs_state < 5:
        mean_GPi = np.mean(recordings["GPi"])
        save_GPi_r(mean_GPi, dbs_state, shortcut, parameter)


#########################################################################################################
########################################### function call ###############################################
#########################################################################################################

simulate()
