import ANNarchy as ann
import BG_model as BG
import numpy as np
import pandas as pd
import sys
import random

########################### fixed random values #################################

initial_seed = int(sys.argv[1]) + 17112023  # number of persons (1-14) + date
# print("\n", initial_seed, "\n")
random.seed(initial_seed)


def trial_seed(trial: int):
    """
    Create unique seed depending on global seed and trial.
    """
    ret = initial_seed + int(f"{trial:0<8}")
    return ret


################################### reset #######################################
def reset_activity():
    "Resets activity in the network by setting all inputs to 0.0"
    BG.IT.B = 0.0
    BG.PPN.B = 0.0
    BG.SNc.alpha = 0.0
    ann.simulate(200.0)


##############################################################################
################################# test run ###################################
##############################################################################


def trial(
    id,
    save_trials,
    condition,
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
):
    reward_setzen = False
    idx1 = 1
    idx2 = 2

    ### save weigths and firing rates + set seed values for every trial
    if save_trials == "True" and k >= 79:
        ann.set_seed(trial_seed(k))
        ann.save(
            f"data/load_simulation_data/save_data/dbs_state_{dbs_state}_person_{id}_trial_{k}.txt"
        )

    ### load weigths and firing rates + set seed values for every trial
    if save_trials == "False":
        ann.load(
            f"data/load_simulation_data/save_data/dbs_state_{dbs_state}_person_{id}_trial_{k}.txt"
        )
        ann.set_seed(trial_seed(k))

    # start value
    IT.B = 0.0
    ann.simulate(100.0)

    # set input
    IT.B = 1.0

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
    # always reward
    if immer:
        if (np.argmax(PFC.r) == 0) or (np.argmax(PFC.r) == 1):
            reward_setzen = True

    # never reward
    if nie:
        if (np.argmax(PFC.r) == 0) or (np.argmax(PFC.r) == 1):
            reward_setzen = False

    # random reward
    if immer == False and nie == False:
        if (np.argmax(PFC.r) == idx1 and reward[k] == 1) or (
            np.argmax(PFC.r) == idx2 and reward[k] == 0
        ):
            reward_setzen = True

    ############################### set reward ###############################
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

    return erfolg


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


def save_data(success, condition):
    success = [int(x) for x in success]

    ############################ save data ##########################

    column = int(sys.argv[1])
    dbs = int(sys.argv[2])
    df = pd.DataFrame(success)

    # filepath
    filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{dbs}_Condition_{condition}.json"
    if column == 0:
        df.to_json(filepath, orient="records", lines=True)
    else:
        data = pd.read_json(filepath, orient="records", lines=True)
        data[column] = success
        data.to_json(filepath, orient="records", lines=True)


##############################################################################
################################ run simulation ##############################
##############################################################################


def simulate():
    id = int(sys.argv[1])
    shortcut = int(sys.argv[3])
    save_trials = sys.argv[6]
    condition = int(sys.argv[7])
    dbs_state = int(sys.argv[2])
    parameter = 0
    dbs_param_state = 0

    ####################### set learn condition ##############################

    if save_trials == "False":

        # DBS-ON learn on
        if condition == 1:
            dbs_state = int(sys.argv[2])

        # simulate-ON / load-ON
        if condition == 2:
            dbs_state = int(sys.argv[2])

        # simulate-OFF / load-ON
        if condition == 3:
            dbs_state = int(sys.argv[2])

        # simulate-ON / load-OFF
        if condition == 4:
            dbs_state = 0

        # simulate-OFF / load-OFF
        if condition == 5:
            dbs_state = 0

    print(
        "\n", "dbs-state:", dbs_state, "simulation:", id, "condition:", condition, "\n"
    )

    ########################## compile BG model ##############################

    populations = BG.create_network(
        initial_seed, dbs_state, shortcut, parameter, dbs_param_state
    )

    ################## Populationsparameter übergeben ########################

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

    # parameter
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
    reward = create_reward(anz_trials, belohnungsverteilung, zufall)

    ############################# initial monitoring ###########################

    monitor = BG.BGMonitor([IT, StrD1, StrD2, STN, GPe, GPi, SNc, Thal, PFC, StrThal])

    ################## simulate 120 trials (1 simulation) ######################

    # start dbs
    if dbs_state > 0:
        DBS.on()

    # option save data: run all 120 trials / option load data: run the last 40 trials
    if save_trials == "True":
        start_trial = 0
    else:
        start_trial = 79

    # run simulation
    for t in range(start_trial, anz_trials):  # number of trials

        # set condition to load a trial
        if save_trials == "False":

            ITStrD1.plasticity = False
            ITStrD2.plasticity = False
            ITSTN.plasticity = False
            StrD1GPi.plasticity = False
            STNGPi.plasticity = False
            StrD1SNc.plasticity = False
            ITThal.plasticity = False

            # DBS-ON learn off
            # if condition == 1:

            # simulate-ON / load-ON
            if condition == 2:
                dbs_state = int(sys.argv[2])

            # simulate-ON / load-OFF
            if condition == 3:
                dbs_state = 0

            # simulate-OFF / load-ON
            if condition == 4:
                dbs_state = int(sys.argv[2])

            # simulate-OFF / load-OFF
            if condition == 5:
                dbs_state = 0

        if t == start_trial:
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
            success.append(
                trial(
                    id,
                    save_trials,
                    condition,
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
                )
            )  # record rewards
        else:
            success.append(
                trial(
                    id,
                    save_trials,
                    condition,
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
                )
            )
        if t == 119:
            recordings = monitor.get()
            monitor.stop()

    if save_trials == "False":
        save_data(success, condition)


#########################################################################################################
############################################ function call ##############################################
#########################################################################################################

simulate()
