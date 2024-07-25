import ANNarchy as ann
import BG_model as BG
import numpy as np
import pandas as pd
import sys
import random

######## fixed random values, set dbs-state and shortcut #############

seed = int(sys.argv[5]) + 17112023  # 17112023 = Datum
random.seed(seed)

##############################################################################
############################### one trial ####################################
##############################################################################


def simulation_data_onetrial(
    IT,
    StrD1,
    StrD2,
    STN,
    GPi,
    GPe,
    Thal,
    PFC,
    StrThal,
    simulation_time,
    boxplots,
):

    # start monitoring
    IT_m = BG.Monitor(IT, ["r"])
    StrD1_m = BG.Monitor(StrD1, ["r"])
    StrD2_m = BG.Monitor(StrD2, ["r"])
    STN_m = BG.Monitor(STN, ["r"])
    GPi_m = BG.Monitor(GPi, ["r"])
    GPe_m = BG.Monitor(GPe, ["r"])
    Thal_m = BG.Monitor(Thal, ["r"])
    PFC_m = BG.Monitor(PFC, ["r"])
    StrThal_m = BG.Monitor(StrThal, ["r"])

    # simulation time
    ann.simulate(simulation_time)

    # get firing rates from 2500ms - 3000ms
    IT_r = IT_m.get("r")
    StrD1_r = StrD1_m.get("r")
    StrD2_r = StrD2_m.get("r")
    STN_r = STN_m.get("r")
    GPi_r = GPi_m.get("r")
    GPe_r = GPe_m.get("r")
    Thal_r = Thal_m.get("r")
    PFC_r = PFC_m.get("r")
    StrThal_r = StrThal_m.get("r")

    # mean firing rates over 500ms
    simulation_data = (
        [
            np.mean(IT_r),
            np.mean(StrD1_r),
            np.mean(StrD2_r),
            np.mean(STN_r),
            np.mean(GPi_r),
            np.mean(GPe_r),
            np.mean(Thal_r),
            np.mean(PFC_r),
            np.mean(StrThal_r),
        ],
    )

    return simulation_data


##############################################################################
############################ simulation run ##################################
##############################################################################


def one_trial(
    IT,
    StrD1,
    StrD2,
    STN,
    GPi,
    GPe,
    Thal,
    PFC,
    StrThal,
    DBS,
    dbs_state,
    session,
    boxplots,
):
    ########################## initial data #####################################
    IT.B = 0.0
    ann.simulate(100)

    ########################## learning time ####################################
    if session == 1 or session == 3:
        IT.B = 1.0
    ann.simulate(2500)

    # start dbs
    if dbs_state > 0 and session > 1:
        DBS.on()
    ########################## learning data ####################################
    simulation_time = 500.0
    data = simulation_data_onetrial(
        IT,
        StrD1,
        StrD2,
        STN,
        GPi,
        GPe,
        Thal,
        PFC,
        StrThal,
        simulation_time,
        boxplots,
    )

    return data


##############################################################################
########################## simulation settings ###############################
##############################################################################


def simulate():
    dbs_state = int(sys.argv[1])
    shortcut = int(sys.argv[2])
    session = int(sys.argv[3])
    boxplots = sys.argv[4]
    parameter = 0
    dbs_param_state = 0

    ######################### compile BG_Modell ##############################

    populations = BG.create_network(
        seed, dbs_state, shortcut, parameter, dbs_param_state
    )

    #################### pass population parameters ##########################

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

    ####################### start simulation ##########################

    # start dbs
    if dbs_state > 0 and session < 2:
        DBS.on()

    data = one_trial(
        IT,
        StrD1,
        StrD2,
        STN,
        GPi,
        GPe,
        Thal,
        PFC,
        StrThal,
        DBS,
        dbs_state,
        session,
        boxplots,
    )

    # save data
    if boxplots == "False":
        filepath = f"data/activity_change/activity_change_dbs_state{dbs_state}_session{session}_id{int(sys.argv[5])}.json"
    if boxplots == "True":
        filepath = f"data/activity_change/activity_change_box_dbs_state{dbs_state}_session{session}_id{int(sys.argv[5])}.json"

    df = pd.DataFrame(data)
    df.to_json(filepath, orient="records", lines=True)


#########################################################################################################
######################################### function call #################################################
#########################################################################################################

simulate()
