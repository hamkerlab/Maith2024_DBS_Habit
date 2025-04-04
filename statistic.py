import numpy as np
import pandas as pd
import pingouin as pg
import itertools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

####################################### copied from fitting_q_learning ##############################################


def transform_range(vector: np.ndarray, new_min, new_max):
    return ((vector - vector.min()) / (vector.max() - vector.min())) * (
        new_max - new_min
    ) + new_min


def load_experimental_data(
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
            DataFrame containing the subject, trial, choice, reward and session columns
    """
    # data needs to be loaded differently for patients/simulations
    if subject_type == "patients":
        file_name = "data/patient_data/choices_rewards_per_trial.pkl"
        # load data using pickle
        with open(file_name, "rb") as f:
            data_patients = pickle.load(f)
        # load data about completed trials per session
        completed = pd.read_json(
            f"data/patient_data/Anz_CompleteTasks_{dbs_state}.json",
            orient="records",
            lines=True,
        )
        completed = completed.to_numpy().astype(int)
        # get the correct format for the data
        ret = {}
        ret["subject"] = []
        ret["trial"] = []
        ret["choice"] = []
        ret["reward"] = []
        ret["session"] = []
        for subject in data_patients:
            for trial, choice in enumerate(
                data_patients[subject][dbs_state]["choices"]
            ):
                ret["subject"].append(subject)
                ret["trial"].append(trial)
                ret["choice"].append(choice)
            for reward in data_patients[subject][dbs_state]["rewards"]:
                ret["reward"].append(reward)
            # column for session (1, 2 or 3)
            ret["session"].append(np.repeat(np.arange(1, 4), completed[subject, :]))

        # concatenate the session column
        ret["session"] = np.concatenate(ret["session"]).tolist()

        # choices should be 0 and 1
        ret["choice"] = (
            transform_range(np.array(ret["choice"]), new_min=0, new_max=1)
            .astype(int)
            .tolist()
        )

    elif subject_type == "simulations":
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
        ret["session"] = []
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
            # column for session (1, 2 or 3), each session has 40 trials
            ret["session"].append(np.repeat(np.arange(1, 4), 40))

        # concatenate the session column
        ret["session"] = np.concatenate(ret["session"]).tolist()

        # choices should be 0 and 1
        ret["choice"] = (
            transform_range(np.array(ret["choice"]), new_min=0, new_max=1)
            .astype(int)
            .tolist()
        )

    return pd.DataFrame(ret)


#####################################################################################################
####################################### ground functions ############################################
#####################################################################################################

####################################### read json data ##############################################


def read_json_data(filepath):
    data = pd.read_json(filepath, orient="records", lines=True)
    data = data.to_numpy()
    return data


##################################### transform json data ###########################################


def read_json_data_T(filepath):
    data = pd.read_json(filepath, orient="records", lines=True)
    data = data.to_numpy()
    data = data.T
    return data


################################### save json and excel data ########################################


def save_table(table, filename):
    df_umkehr = pd.DataFrame(table)
    filepath = f"{filename}.json"
    df_umkehr.to_json(filepath, orient="records", lines=True)
    filepath = f"{filename}.xlsx"
    df_umkehr.to_excel(filepath, index=False)


#############################################################################################################
############################################ normalize data #################################################
#############################################################################################################


def normalize_data(data):
    result = [(data - min(data)) / (max(data) - min(data))]

    # result = data + min(data)
    # result = result / max(result)
    # result = np.round(result, 3)
    return result[0]


#############################################################################################################
###################################### calculate averages for 3 seessions ###################################
#############################################################################################################


def mean_data(result):
    number_result = len(result)

    if number_result > 1:
        mean = [
            np.nanmean(result[:, 0], axis=0),
            np.nanmean(result[:, 1], axis=0),
            np.nanmean(result[:, 2], axis=0),
        ]
    else:
        mean = [
            result[0][0],
            result[0][1],
            result[0][2],
        ]

    return mean


#############################################################################################################
###################################### calculate averages for 5 trials ######################################
#############################################################################################################


def mean_data_line(result):
    number_result = len(result)

    if number_result > 1:
        # mean of 24 colums (axis=0)
        mean = np.nanmean(result, axis=0)
    else:
        mean = result[0]

    return mean


#############################################################################################################
############################# calculate averages (patient data) for 3 sessions ##############################
#############################################################################################################


def mean_dbs_session(filepath1, filepath2, habit):
    ############################################# read data ##########################################

    DataON = read_json_data(filepath1)
    DataOFF = read_json_data(filepath2)

    if habit:
        DataON = 40 - DataON
        DataOFF = 40 - DataOFF

    ################# mean session 3 for every dbs_state ####################

    mean1_ON = np.nanmean(DataON[:, 0], axis=0)
    mean2_ON = np.nanmean(DataON[:, 1], axis=0)
    mean3_ON = np.nanmean(DataON[:, 2], axis=0)

    mean1_OFF = np.nanmean(DataOFF[:, 0], axis=0)
    mean2_OFF = np.nanmean(DataOFF[:, 1], axis=0)
    mean3_OFF = np.nanmean(DataOFF[:, 2], axis=0)

    mean = [[mean1_ON, mean2_ON, mean3_ON], [mean1_OFF, mean2_OFF, mean3_OFF]]

    return mean


#############################################################################################################
######################################### calculate standard error ##########################################
#############################################################################################################


def standarderror_data(result):
    number_result = len(result)

    if number_result > 1:
        standarderror = [
            np.nanstd(result[:, 0], axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(result[:, 0]), axis=0)),
            np.nanstd(result[:, 1], axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(result[:, 1]), axis=0)),
            np.nanstd(result[:, 2], axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(result[:, 2]), axis=0)),
        ]
    else:
        standarderror = [0, 0, 0]

    return standarderror


#############################################################################################################
######################################### calculate standard deviation ######################################
#############################################################################################################


def standarddeviation_data(result):
    number_result = len(result)

    if number_result > 1:
        standarddeviation = [
            np.nanstd(result[:, 0], axis=0, ddof=1),
            np.nanstd(result[:, 1], axis=0, ddof=1),
            np.nanstd(result[:, 2], axis=0, ddof=1),
        ]
    else:
        standarddeviation = [0, 0, 0]

    return standarddeviation


#############################################################################################################
############################### calculate standard error session 3 ##########################################
#############################################################################################################


def standarderror_session3(result):

    number_result = len(result)

    if number_result > 1:
        standarderror = [
            np.nanstd(result[:, 0], axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(result[:, 0]), axis=0)),
        ]
    else:
        standarderror = [0]

    return standarderror


#####################################################################################################
############################  calculate standard error patient data #################################
#####################################################################################################


def session_standard_error(filepath1, filepath2, habit):
    ######################################## read data #############################################

    DataON = read_json_data(filepath1)
    DataOFF = read_json_data(filepath2)

    if habit:
        DataON = 40 - DataON
        DataOFF = 40 - DataOFF

    ######################## error for every session ##################################

    standarderrorON = [
        np.nanstd(DataON[:, 0], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataON[:, 0]), axis=0)),
        np.nanstd(DataON[:, 1], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataON[:, 1]), axis=0)),
        np.nanstd(DataON[:, 2], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataON[:, 2]), axis=0)),
    ]

    standarderrorOFF = [
        np.nanstd(DataOFF[:, 0], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataOFF[:, 0]), axis=0)),
        np.nanstd(DataOFF[:, 1], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataOFF[:, 1]), axis=0)),
        np.nanstd(DataOFF[:, 2], axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(DataOFF[:, 2]), axis=0)),
    ]

    standarderror = [standarderrorON, standarderrorOFF]

    return standarderror


#############################################################################################################
################################ total rewards (simulation data) per session ################################
#############################################################################################################


def processing_data(data, number_of_persons):

    number_data = len(data[0])
    if number_data < number_of_persons:
        number_of_persons = number_data

    for i in range(number_of_persons):
        reward_session1 = np.sum(data[:40, i])
        reward_session2 = np.sum(data[40:80, i])
        reward_session3 = np.sum(data[80:120, i])

        reward_sessions = np.array([reward_session1, reward_session2, reward_session3])

        if i == 0:
            result = [reward_sessions]
        else:
            result = np.vstack([result, reward_sessions])

    return result


#############################################################################################################
########################## total unrewarded decisions (simulation data) per session #########################
#############################################################################################################


def processing_habit_data(data, number_of_persons):
    """
    Returns:
        result (np.array):
            Array with the number of unrewarded decisions for each session per person
            with shape (number_of_persons, 3).
    """

    number_data = len(data[0])
    if number_data < number_of_persons:
        number_of_persons = number_data

    for i in range(number_of_persons):
        reward_session1 = np.sum(data[:40, i])
        reward_session2 = np.sum(data[40:80, i])
        reward_session3 = np.sum(data[80:120, i])

        habit_sessions = np.array(
            [40 - reward_session1, 40 - reward_session2, 40 - reward_session3]
        )

        if i == 0:
            result = [habit_sessions]
        else:
            result = np.vstack([result, habit_sessions])

    return result


#############################################################################################################
################################# total unrewarded decisions for 5 trials ###################################
#############################################################################################################


def processing_line(data, number_of_persons):

    number_data = len(data[0])
    if number_data < number_of_persons:
        number_of_persons = number_data

    result = []

    for i in range(number_of_persons):
        habit_sessions = []

        # 24 sessions, 5 trials per session
        for j in range(24):
            # start- and endindex
            start_idx = j * 5
            end_idx = start_idx + 5

            # total 5 trials
            reward_sum = np.sum(data[start_idx:end_idx, i])

            # unrewarded decitions
            adjusted_sum = 5 - reward_sum
            habit_sessions.append(adjusted_sum)

        if i == 0:
            result = np.array([habit_sessions])
        else:
            result = np.vstack([result, habit_sessions])

    return result


#############################################################################################################
############################## total rewards (parameter data) per session ###################################
#############################################################################################################


def processing_data_param(data):

    number_data = len(data[0])

    for i in range(number_data):
        reward_session1 = np.sum(data[:40, i])
        reward_session2 = np.sum(data[40:80, i])
        reward_session3 = np.sum(data[80:120, i])

        reward_sessions = np.array([reward_session1, reward_session2, reward_session3])

        if i == 0:
            result = [reward_sessions]
        else:
            result = np.vstack([result, reward_sessions])

    return result


#############################################################################################################
################################# performance simulation data/patient data ##################################
#############################################################################################################


def processing_performance_param(data):

    ################################### data  De A Marcelino DBS-ON ###############################
    filepath1 = "data/patient_data/RewardsPerSession_ON.json"
    filepath2 = "data/patient_data/RewardsPerSession_OFF.json"

    mean = mean_dbs_session(filepath1, filepath2, False)
    mean = mean[0]
    performance = [data[0] / mean[0], data[1] / mean[1], data[2] / mean[2]]

    return performance


#############################################################################################################
############################# total rewards (load simulation data) in session 3 #############################
#############################################################################################################


def processing_data_session3(data, number_of_persons):

    number_data = len(data[0])
    if number_data < number_of_persons:
        number_of_persons = number_data

    for i in range(number_of_persons):
        reward_session3 = np.sum(data[:40, i])

        if i == 0:
            result = [reward_session3]
        else:
            result = np.vstack([result, reward_session3])

    return result


#############################################################################################################
###################### total unrewarded decisions (load simulation data) in session 3 #######################
#############################################################################################################


def processing_habit_session3(data, number_of_persons):
    ####################################### total over 3 sessions ##################################

    number_data = len(data[0])
    if number_data < number_of_persons:
        number_of_persons = number_data

    for i in range(number_of_persons):
        reward_session3 = np.sum(data[:40, i])

        habit_sessions = np.array([40 - reward_session3])

        if i == 0:
            result = [habit_sessions]
        else:
            result = np.vstack([result, habit_sessions])

    return result


#####################################################################################################
################################### save means and errors ###########################################
#####################################################################################################


def save_mean_error(number_of_persons):

    ######################################## read data ############################################

    data01 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State0.json")
    data02 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State1.json")
    data03 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State2.json")
    data04 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State3.json")
    data05 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State4.json")
    data06 = read_json_data("data/simulation_data/Results_Shortcut0_DBS_State5.json")

    data11 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State0.json")
    data12 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State1.json")
    data13 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State2.json")
    data14 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State3.json")
    data15 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State4.json")
    data16 = read_json_data("data/simulation_data/Results_Shortcut1_DBS_State5.json")

    ####################################### processing data #######################################

    # mean shortcut off
    data01 = 40 - processing_data(data01, number_of_persons)
    data02 = 40 - processing_data(data02, number_of_persons)
    data03 = 40 - processing_data(data03, number_of_persons)
    data04 = 40 - processing_data(data04, number_of_persons)
    data05 = 40 - processing_data(data05, number_of_persons)
    data06 = 40 - processing_data(data06, number_of_persons)

    mean_shortoff1 = mean_data(data01)
    mean_shortoff2 = mean_data(data02)
    mean_shortoff3 = mean_data(data03)
    mean_shortoff4 = mean_data(data04)
    mean_shortoff5 = mean_data(data05)
    mean_shortoff6 = mean_data(data06)
    error_shortoff1 = standarddeviation_data(data01)
    error_shortoff2 = standarddeviation_data(data02)
    error_shortoff3 = standarddeviation_data(data03)
    error_shortoff4 = standarddeviation_data(data04)
    error_shortoff5 = standarddeviation_data(data05)
    error_shortoff6 = standarddeviation_data(data06)

    # mean shortcut on
    data11 = 40 - processing_data(data11, number_of_persons)
    data12 = 40 - processing_data(data12, number_of_persons)
    data13 = 40 - processing_data(data13, number_of_persons)
    data14 = 40 - processing_data(data14, number_of_persons)
    data15 = 40 - processing_data(data15, number_of_persons)
    data16 = 40 - processing_data(data16, number_of_persons)

    mean_shorton1 = mean_data(data11)
    mean_shorton2 = mean_data(data12)
    mean_shorton3 = mean_data(data13)
    mean_shorton4 = mean_data(data14)
    mean_shorton5 = mean_data(data15)
    mean_shorton6 = mean_data(data16)
    error_shorton1 = standarddeviation_data(data11)
    error_shorton2 = standarddeviation_data(data12)
    error_shorton3 = standarddeviation_data(data13)
    error_shorton4 = standarddeviation_data(data14)
    error_shorton5 = standarddeviation_data(data15)
    error_shorton6 = standarddeviation_data(data16)

    ####################################### save tables #######################################

    # Shortcut OFF
    table_shortcutoff = {
        "learning phase": [
            "session1",
            "session2",
            "session3",
        ],
        "dbs-off mean": [
            mean_shortoff1[0],
            mean_shortoff1[1],
            mean_shortoff1[2],
        ],
        "dbs-off standard deviation": [
            error_shortoff1[0],
            error_shortoff1[1],
            error_shortoff1[2],
        ],
        "suppression mean": [
            mean_shortoff2[0],
            mean_shortoff2[1],
            mean_shortoff2[2],
        ],
        "suppression standard deviation": [
            error_shortoff2[0],
            error_shortoff2[1],
            error_shortoff2[2],
        ],
        "efferent mean": [
            mean_shortoff3[0],
            mean_shortoff3[1],
            mean_shortoff3[2],
        ],
        "efferent standard deviation": [
            error_shortoff3[0],
            error_shortoff3[1],
            error_shortoff3[2],
        ],
        "afferent mean": [
            mean_shortoff4[0],
            mean_shortoff4[1],
            mean_shortoff4[2],
        ],
        "afferent standard deviation": [
            error_shortoff4[0],
            error_shortoff4[1],
            error_shortoff4[2],
        ],
        "passing-fibres mean": [
            mean_shortoff5[0],
            mean_shortoff5[1],
            mean_shortoff5[2],
        ],
        "passing-fibres standard deviation": [
            error_shortoff5[0],
            error_shortoff5[1],
            error_shortoff5[2],
        ],
        "dbs-comb mean": [
            mean_shortoff6[0],
            mean_shortoff6[1],
            mean_shortoff6[2],
        ],
        "dbs-comb standard deviation": [
            error_shortoff6[0],
            error_shortoff6[1],
            error_shortoff6[2],
        ],
    }

    # round all values
    for i in table_shortcutoff:
        if i != "learning phase":
            table_shortcutoff[i] = [
                np.round(value, 2) for value in table_shortcutoff[i]
            ]
    save_table(
        table_shortcutoff,
        f"statistic/means_and_erros_shortcutoff_{number_of_persons}",
    )

    # Shortcut ON
    table_shortcuton = {
        "learning phase": [
            "session1",
            "session2",
            "session3",
        ],
        "dbs-off mean": [
            mean_shorton1[0],
            mean_shorton1[1],
            mean_shorton1[2],
        ],
        "dbs-off standard deviation": [
            error_shorton1[0],
            error_shorton1[1],
            error_shorton1[2],
        ],
        "suppression mean": [
            mean_shorton2[0],
            mean_shorton2[1],
            mean_shorton2[2],
        ],
        "suppression standard deviation": [
            error_shorton2[0],
            error_shorton2[1],
            error_shorton2[2],
        ],
        "efferent mean": [
            mean_shorton3[0],
            mean_shorton3[1],
            mean_shorton3[2],
        ],
        "efferent standard deviation": [
            error_shorton3[0],
            error_shorton3[1],
            error_shorton3[2],
        ],
        "afferent mean": [
            mean_shorton4[0],
            mean_shorton4[1],
            mean_shorton4[2],
        ],
        "afferent standard deviation": [
            error_shorton4[0],
            error_shorton4[1],
            error_shorton4[2],
        ],
        "passing-fibres mean": [
            mean_shorton5[0],
            mean_shorton5[1],
            mean_shorton5[2],
        ],
        "passing-fibres standard deviation": [
            error_shorton5[0],
            error_shorton5[1],
            error_shorton5[2],
        ],
        "dbs-comb mean": [
            mean_shorton6[0],
            mean_shorton6[1],
            mean_shorton6[2],
        ],
        "dbs-comb standard deviation": [
            error_shorton6[0],
            error_shorton6[1],
            error_shorton6[2],
        ],
    }

    # round all values
    for i in table_shortcuton:
        if i != "learning phase":
            table_shortcuton[i] = [np.round(value, 2) for value in table_shortcuton[i]]
    save_table(
        table_shortcuton,
        f"statistic/means_and_erros_shortcuton_{number_of_persons}",
    )


#####################################################################################################
##################################### check normal distribution #####################################
#####################################################################################################


def normality(data):
    # Shapiro-Wilk Test
    result = pg.normality(data)

    return result["normal"].values[0]


#####################################################################################################
################################### check homogeneity of variances ##################################
#####################################################################################################


def homo_var(data):
    # Levene-Test
    result = pg.homoscedasticity(data)
    return result["equal_var"].values[0]


#####################################################################################################
############################################# T-Test ################################################
#####################################################################################################


def ttest_data(data1, data2):
    ######################################## processing data #######################################

    mask = np.isnan(data1)
    data1 = data1[~mask]
    data1 = [int(x) for x in data1]

    mask = np.isnan(data2)
    data2 = data2[~mask]
    data2 = [int(x) for x in data2]

    ######################################## check sphericity #######################################
    norm1 = normality(data1)
    norm2 = normality(data2)

    if norm1 == True and norm2 == True:
        normal = True
    else:
        normal = False

    varhomo_data = [data1, data2]
    varhomo = homo_var(varhomo_data)

    ############################################# T-Test ##############################################

    t_statistic = pg.ttest(data1, data2, correction=True)

    t_value = t_statistic["T"].values[0]
    p_value = t_statistic["p-val"].values[0]
    df = t_statistic["dof"].values[0]

    # show results
    print("t(", df, "):", t_value)
    print("p:", p_value)
    print("normality:", normal)
    print("homogeneity of variances:", varhomo)

    ######################################### significanz yes/no #######################################

    # interpret results
    if p_value < 0.05:
        difference = 1
        print(
            "There is a significant difference between the groups.",
            "\n",
        )
    else:
        difference = 0
        print(
            "There is no significant difference between the groups.",
            "\n",
        )

    ttest = [t_value, p_value, df, difference, normal, varhomo]
    return ttest


#####################################################################################################
############################################### ANOVA ###############################################
#####################################################################################################


def anova_group(*values):
    ######################################## processing data #######################################

    data = []

    for i, data_i in enumerate(values):
        mask = np.isnan(data_i)
        data_i = data_i[~mask]
        data_i = [int(x) for x in data_i]
        data.append(np.array(data_i))

    print(data)

    ######################################## check sphericity #######################################

    normal = True

    for i, norm_i in enumerate(data):
        norm = normality(norm_i)
        if norm == False:
            normal = False

    varhomo = homo_var(data)

    ################################ prepare pandas dataframe for ANOVA ##############################

    dataframe = np.concatenate(data)
    group = []

    for i, data_i in enumerate(data):
        length = len(data_i)
        groupname = [f"Group{i}" for j in range(length)]
        group = group + groupname

    pandas_dataframe = pd.DataFrame({"Value": dataframe, "Group": group})

    # print(pandas_dataframe)
    print(" ")

    #################################################### ANOVA  ####################################################

    f_statistic = pg.welch_anova(data=pandas_dataframe, dv="Value", between="Group")
    f_value = f_statistic["F"].values[0]
    p_value = f_statistic["p-unc"].values[0]
    df_z = f_statistic["ddof1"].values[0]
    df_n = f_statistic["ddof2"].values[0]

    # print results
    print("F(", df_z, ",", df_n, "):", f_value)
    print("p:", p_value)
    print("normality:", normal)
    print("homogeneity of variances:", varhomo)

    # interpret results
    if p_value < 0.05:
        difference = 1
        print(
            "There is a significant difference between the groups.",
            "\n",
        )
    else:
        difference = 0
        print(
            "There is no significant difference between the groups.",
            "\n",
        )

    anova = [f_value, p_value, df_z, df_n, difference, normal, varhomo]
    return anova


def anova_group2(*values):
    ######################################## processing data #######################################

    data = []

    for i, data_i in enumerate(values):
        # mask = np.isnan(data_i)
        # data_i = data_i[~mask]
        # data_i = [int(x) for x in data_i]
        data.append(np.array(data_i))

    # print(data)

    ######################################## check sphericity #######################################

    normal = True

    for i, norm_i in enumerate(data):
        norm = normality(norm_i)
        if norm == False:
            normal = False

    varhomo = homo_var(data)

    ################################ prepare pandas dataframe for ANOVA ##############################

    dataframe = np.concatenate(data)
    group = []

    for i, data_i in enumerate(data):
        length = len(data_i)
        groupname = [f"Group{i}" for j in range(length)]
        group = group + groupname

    pandas_dataframe = pd.DataFrame({"Value": dataframe, "Group": group})

    # print(pandas_dataframe)
    # print(" ")

    #################################################### ANOVA  ####################################################

    f_statistic = pg.welch_anova(data=pandas_dataframe, dv="Value", between="Group")
    f_value = f_statistic["F"].values[0]
    p_value = f_statistic["p-unc"].values[0]
    df_z = f_statistic["ddof1"].values[0]
    df_n = f_statistic["ddof2"].values[0]

    # print results
    print("F(", df_z, ",", df_n, "):", f_value)
    print("p:", p_value)
    print("normality:", normal)
    print("homogeneity of variances:", varhomo)

    # interpret results
    if p_value < 0.05:
        difference = 1
        print(
            "There is a significant difference between the groups.",
            "\n",
        )
    else:
        difference = 0
        print(
            "There is no significant difference between the groups.",
            "\n",
        )

    anova = [f_value, p_value, df_z, df_n, difference, normal, varhomo]
    return anova


#####################################################################################################
################################### repeaded two-sided ANOVA ########################################
#####################################################################################################


def rm_anova_group(*values):
    ######################################## processing data #######################################

    data = []

    dbs_state = values[-1]
    values = values[:-1]

    for i, data_i in enumerate(values):
        mask = np.isnan(data_i)
        data_i = data_i[~mask]
        data_i = [int(x) for x in data_i]
        data.append(np.array(data_i))

    ######################################## check sphericity #######################################

    normal = True

    for i, norm_i in enumerate(data):
        norm = normality(norm_i)
        if norm == False:
            normal = False

    varhomo = homo_var(data)

    ############################## prepare pandas dataframe for ANOVA ################################

    dataframe = np.concatenate(data)

    simulation_id = []
    simulateDBS = []
    loadDBS = []

    for j in range(len(data)):
        for i in range(100):
            simulation_id.append(i)

            if j == 0:
                simulateDBS.append("on")
                loadDBS.append("on")
            if j == 1:
                simulateDBS.append("on")
                loadDBS.append("off")
            if j == 2:
                simulateDBS.append("off")
                loadDBS.append("on")
            if j == 3:
                simulateDBS.append("off")
                loadDBS.append("off")

    pandas_dataframe = pd.DataFrame(
        {
            "simID": simulation_id,
            "unrewarded_decisions": dataframe,
            "simulateDBS": simulateDBS,
            "loadDBS": loadDBS,
        }
    )

    #################################################### ANOVA  ####################################################

    f_statistic = pg.rm_anova(
        data=pandas_dataframe,
        dv="unrewarded_decisions",
        within=["simulateDBS", "loadDBS"],
        subject="simID",
    )
    print("\n", f_statistic, "\n")

    return f_statistic


#####################################################################################################
############################################## paired T-Test ########################################
#####################################################################################################


def pairwise_ttest(*values):
    ######################################## processing data #######################################

    data = []

    dbs_state = values[-1]
    values = values[:-1]

    for i, data_i in enumerate(values):
        mask = np.isnan(data_i)
        data_i = data_i[~mask]
        data_i = [int(x) for x in data_i]
        data.append(np.array(data_i))

    ######################################## check sphericity #######################################

    normal = True

    for i, norm_i in enumerate(data):
        norm = normality(norm_i)
        if norm == False:
            normal = False

    varhomo = homo_var(data)

    ############################## prepare pandas dataframe for T-Test ################################

    dataframe = np.concatenate(data)

    simulation_id = []
    simulateDBS = []
    loadDBS = []

    for j in range(len(data)):
        for i in range(100):
            simulation_id.append(i)

            if j == 0:
                simulateDBS.append("on")
                loadDBS.append("on")
            if j == 1:
                simulateDBS.append("on")
                loadDBS.append("off")
            if j == 2:
                simulateDBS.append("off")
                loadDBS.append("on")
            if j == 3:
                simulateDBS.append("off")
                loadDBS.append("off")

    pandas_dataframe = pd.DataFrame(
        {
            "simID": simulation_id,
            "unrewarded_decisions": dataframe,
            "simulateDBS": simulateDBS,
            "loadDBS": loadDBS,
        }
    )

    ############################################ paired T-Test ##############################################

    t_statistic1 = pg.pairwise_tests(
        data=pandas_dataframe,
        dv="unrewarded_decisions",
        within=["loadDBS", "simulateDBS"],
        subject="simID",
        parametric=True,
        interaction=True,
    )
    t_statistic2 = pg.pairwise_tests(
        data=pandas_dataframe,
        dv="unrewarded_decisions",
        within=["simulateDBS", "loadDBS"],
        subject="simID",
        parametric=True,
        interaction=True,
    )

    t_statistic1.rename(columns={"loadDBS": "First of Interaction"}, inplace=True)
    t_statistic2.rename(columns={"simulateDBS": "First of Interaction"}, inplace=True)
    print("\n", t_statistic1, "\n")
    print("\n", t_statistic2, "\n")
    t_statistic = pd.concat([t_statistic1, t_statistic2.tail(2)], ignore_index=True)

    print("\n", t_statistic, "\n")

    return t_statistic


#####################################################################################################
############################################## dbs_on_vs_off ########################################
#####################################################################################################


def dbs_on_vs_off(number_of_persons=None, shortcut=True):
    """
    Compare the unrewarded decisions of the different DBS states to the baseline
    (dbs-off) using a linear mixed effect model. Then compare the different DBS
    states (excluding afferent and dbs-off) in session 3 using a repeated measures
    ANOVA.

    Args:
        number_of_persons (int):
            Number of simulation to consider. If None use patient data.
        shortcut (bool):
            If True use plastic shortcut data, otherwise use the fixed shortcut data.

    Returns:
        data_df_full (pd.DataFrame):
            DataFrame with columns:
                "subject"
                "unrewarded_decisions"
                "dbs_state"
                "session"
    """
    if number_of_persons is not None:
        # Get data in long format with the columns:
        #   "subject"
        #   "unrewarded_decisions"
        #   "dbs_state" (dbs-off, suppression, efferent, afferent, passing fibers, dbs-comb)
        #   "session" (1, 2, 3)
        data_dict = {
            "subject": [],
            "unrewarded_decisions": [],
            "dbs_state": [],
            "session": [],
        }

        # for each dbs state load data and add it to the dictionary
        for dbs_state_id, dbs_state_name in [
            [0, "dbs-off"],
            [1, "suppression"],
            [2, "efferent"],
            [3, "afferent"],
            [4, "passing fibers"],
            [5, "dbs-comb"],
        ]:
            # load the rewarded decisions data
            rewarded_decisions_data = read_json_data(
                f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State{dbs_state_id}.json"
            )

            # convert to array with unrewarded decisions per session
            unrewarded_per_session = processing_habit_data(
                rewarded_decisions_data, number_of_persons
            )

            # loop over subjects, here we only use simulation data therefore we have
            # the same subject ids for each dbs type
            for subject_id, unrewarded_decisions in enumerate(unrewarded_per_session):
                # loop over sessions
                for session_id, unrewarded_decisions_session in enumerate(
                    unrewarded_decisions
                ):
                    # add data to the dictionary
                    data_dict["subject"].append(subject_id)
                    data_dict["unrewarded_decisions"].append(
                        unrewarded_decisions_session
                    )
                    data_dict["dbs_state"].append(dbs_state_name)
                    data_dict["session"].append(session_id + 1)

        # convert dictionary to pandas dataframe
        data_df_full = pd.DataFrame(data_dict)

    else:
        # create the same dataframe for the patient data
        selections_patients_OFF_df = load_experimental_data(
            subject_type="patients",
            shortcut_type=None,
            dbs_state="OFF",
            dbs_variant=None,
        )
        selections_patients_ON_df = load_experimental_data(
            subject_type="patients",
            shortcut_type=None,
            dbs_state="ON",
            dbs_variant=None,
        )
        # combine both dataframes with a new column "dbs_state"
        selections_patients_OFF_df["dbs_state"] = "dbs-off"
        selections_patients_ON_df["dbs_state"] = "dbs-on"
        selections_patients_df = pd.concat(
            [selections_patients_OFF_df, selections_patients_ON_df]
        )
        # add the new column "unrewarded decisions" to the dataframe (1-reward column)
        selections_patients_df["unrewarded_decisions"] = (
            1 - selections_patients_df["reward"]
        )
        # now sum up the unrewarded decisions for each session and subject over all trials of the session
        data_df_full = (
            selections_patients_df.groupby(["subject", "session", "dbs_state"])
            .sum()
            .reset_index()
        )

    # run linear mixed effect model for each session
    # fixed effect for dbs_state compared to baseline (dbs-off)
    # exclude afferent
    for session in [1, 2, 3]:
        data_df = data_df_full.copy()
        data_df = data_df[data_df["session"] == session]
        data_df = data_df[data_df["dbs_state"] != "afferent"]
        data_df["dbs_state"] = data_df["dbs_state"].astype("category")
        data_df["session"] = data_df["session"].astype("category")
        model = smf.mixedlm(
            "unrewarded_decisions ~ C(dbs_state, Treatment('dbs-off'))",
            data_df,
            groups=data_df["subject"],
        )
        result = model.fit()

        # get the p-values for the coefficients and correct them for multiple comparisons
        p_values = result.pvalues
        # exlude Group Var and Intercept
        p_values = p_values.drop("Group Var")
        p_values = p_values.drop("Intercept")
        p_values_corrected = multipletests(p_values, method="bonferroni")[:2]
        p_values_corrected_df = pd.DataFrame(
            {
                "p-values uncor": p_values,
                "p-values cor": p_values_corrected[1],
                "reject": p_values_corrected[0],
            }
        )
        # also add the corresponding  coefficients, Std.Err., z values, and confidence intervalls to the dataframe
        further_columns = {
            "coefficients": result.params,
            "Std.Err.": result.bse,
            "z values": result.tvalues,
            "[0.025": result.conf_int()[0],
            "0.975]": result.conf_int()[1],
        }
        for column_name, column_data in further_columns.items():
            further_columns[column_name] = column_data.drop("Group Var")
            further_columns[column_name] = further_columns[column_name].drop(
                "Intercept"
            )
        p_values_corrected_df = pd.concat(
            [p_values_corrected_df, pd.DataFrame(further_columns)], axis=1
        )

        # save the p_values_corrected_df as csv
        p_values_corrected_df.round(3).to_csv(
            f"statistic/simulation_data_difference_dbs_on_off_{number_of_persons if number_of_persons is not None else 'patients'}_shortcut{int(shortcut)}_session_{session}.csv",
            decimal=",",
        )

        # save results
        with open(
            f"statistic/simulation_data_difference_dbs_on_off_{number_of_persons if number_of_persons is not None else 'patients'}_shortcut{int(shortcut)}_session_{session}.txt",
            "w",
        ) as fh:
            fh.write(result.summary().as_text())
            fh.write("\n")
            fh.write(p_values_corrected_df.round(3).to_string())

        # Create a plot for repeated measures with lines for each subject
        # ignore index in data_df

        # create subplots for each dbs state
        _, ax = plt.subplots(
            nrows=1,
            ncols=len(data_df["dbs_state"].unique()) - 1,
            figsize=(len(data_df["dbs_state"].unique()) * 3, 5),
        )
        for i, dbs_state in enumerate(
            data_df[data_df["dbs_state"] != "dbs-off"]["dbs_state"].unique()
        ):
            # only contain the dbs state and dbs-off
            data_df_subplot = data_df.copy()
            data_df_subplot = data_df_subplot[
                data_df_subplot["dbs_state"].isin(["dbs-off", dbs_state])
            ]
            # order the dbs states
            dbs_state_names_in_df = data_df_subplot["dbs_state"].unique()
            # remove dbs-off from the list
            dbs_state_names_in_df = dbs_state_names_in_df[
                dbs_state_names_in_df != "dbs-off"
            ]
            # prepend dbs-off to the list
            order = np.insert(dbs_state_names_in_df, 0, "dbs-off")
            data_df_subplot["dbs_state"] = pd.Categorical(
                data_df_subplot["dbs_state"], categories=order, ordered=True
            )
            if number_of_persons is None or number_of_persons == 14:
                data_df_subplot["subject"] = data_df_subplot["subject"].astype(
                    "category"
                )
                data_df_subplot = data_df_subplot.reset_index(drop=True)
                sns.lineplot(
                    data=data_df_subplot,
                    x="dbs_state",
                    y="unrewarded_decisions",
                    hue="subject",
                    style="subject",
                    ax=ax[i] if len(data_df["dbs_state"].unique()) > 2 else ax,
                    legend=False,
                )
                # additionally add a barplot with se as error bars
                sns.barplot(
                    data=data_df_subplot,
                    x="dbs_state",
                    y="unrewarded_decisions",
                    ax=ax[i] if len(data_df["dbs_state"].unique()) > 2 else ax,
                    errorbar=("se", 1),
                )
            else:
                sns.boxplot(
                    data=data_df_subplot,
                    x="dbs_state",
                    y="unrewarded_decisions",
                    ax=ax[i] if len(data_df["dbs_state"].unique()) > 2 else ax,
                )
        plt.tight_layout()
        plt.savefig(
            f"statistic/simulation_data_difference_dbs_on_off_{number_of_persons if number_of_persons is not None else 'patients'}_shortcut{int(shortcut)}_session_{session}.png"
        )
        plt.close()

    if number_of_persons is not None:
        # for session 3 compare the different dbs states (excluding afferent and dbs-off)
        # using a repeated measures ANOVA using pingouin rm_anova
        data_df = data_df_full.copy()
        data_df = data_df[data_df["session"] == 3]
        data_df = data_df[data_df["dbs_state"] != "afferent"]
        data_df = data_df[data_df["dbs_state"] != "dbs-off"]
        data_df["dbs_state"] = data_df["dbs_state"].astype("category")
        data_df["session"] = data_df["session"].astype("category")
        aov = pg.rm_anova(
            data=data_df,
            dv="unrewarded_decisions",
            within="dbs_state",
            subject="subject",
            detailed=True,
        )

        # save results
        with open(
            f"statistic/simulation_data_difference_dbs_variants_{number_of_persons}_shortcut{int(shortcut)}_session_3.txt",
            "w",
        ) as fh:
            fh.write(aov.round(3).to_string())

    return data_df_full


#####################################################################################################
####################################### check hypothesis ############################################
#####################################################################################################


def run_statistic(H1, H2, H3, number_of_persons):
    if H1:
        dbs_state = [
            "DBS-OFF (SHORTCUT_ON vs. SHORTCUT_OFF):",
            "suppression (SHORTCUT_ON vs. SHORTCUT_OFF):",
            "EFFERENT (SHORTCUT_ON vs. SHORTCUT_OFF):",
            "AFFERENT (SHORTCUT_ON vs. SHORTCUT_OFF):",
            "PASSING FIBERS (SHORTCUT_ON vs. SHORTCUT_OFF):",
            "DBS_ALL (SHORTCUT_ON vs. SHORTCUT_OFF):",
        ]

        table_H1 = {
            "ShortcutOn vs. ShortcutOFF": [
                "dbs-off",
                "suppression",
                "efferent",
                "afferent",
                "passing fibers",
                "dbs-comb",
            ],
            "S1 t": [],
            "S1 p": [],
            "S1 df": [],
            "S1 diff": [],
            "S1 norm": [],
            "S1 var": [],
            "S2 t": [],
            "S2 p": [],
            "S2 df": [],
            "S2 diff": [],
            "S2 norm": [],
            "S2 var": [],
            "S3 t": [],
            "S3 p": [],
            "S3 df": [],
            "S3 diff": [],
            "S3 norm": [],
            "S3 var": [],
        }

        for i in range(6):
            ########################## load data ################################

            DataBA_1 = read_json_data(
                f"data/simulation_data/Results_Shortcut0_DBS_State{i}.json"
            )
            DataBA_1 = processing_habit_data(DataBA_1, number_of_persons)

            DataBA_2 = read_json_data(
                f"data/simulation_data/Results_Shortcut1_DBS_State{i}.json"
            )
            DataBA_2 = processing_habit_data(DataBA_2, number_of_persons)

            print("\n", DataBA_1, "\n")
            print("\n", DataBA_2, "\n")

            ########################## T-Test ###################################
            print("#############################################")
            print(dbs_state[i], "\n")
            # session1
            Data1 = DataBA_1[:, 0]
            Data2 = DataBA_2[:, 0]
            print("Session 1:")
            ttest = ttest_data(Data1, Data2)
            table_H1["S1 t"].append(ttest[0])
            table_H1["S1 p"].append(ttest[1])
            table_H1["S1 df"].append(ttest[2])
            table_H1["S1 diff"].append(ttest[3])
            table_H1["S1 norm"].append(ttest[4])
            table_H1["S1 var"].append(ttest[5])

            # session2
            Data3 = DataBA_1[:, 1]
            Data4 = DataBA_2[:, 1]
            print("Session 2:")
            ttest = ttest_data(Data3, Data4)
            table_H1["S2 t"].append(ttest[0])
            table_H1["S2 p"].append(ttest[1])
            table_H1["S2 df"].append(ttest[2])
            table_H1["S2 diff"].append(ttest[3])
            table_H1["S2 norm"].append(ttest[4])
            table_H1["S2 var"].append(ttest[5])

            # session3
            Data5 = DataBA_1[:, 2]
            Data6 = DataBA_2[:, 2]
            print("Session 3:")
            ttest = ttest_data(Data5, Data6)
            table_H1["S3 t"].append(ttest[0])
            table_H1["S3 p"].append(ttest[1])
            table_H1["S3 df"].append(ttest[2])
            table_H1["S3 diff"].append(ttest[3])
            table_H1["S3 norm"].append(ttest[4])
            table_H1["S3 var"].append(ttest[5])

        save_table(
            table_H1,
            f"statistic/simulation_data_difference_shortcut_on_off_{number_of_persons}",
        )

    if H2:

        #####################################################################################################
        #################################### !!!! FIG DBS ON OFF !!!!! ######################################
        #####################################################################################################
        print("stats H2 with n = ", number_of_persons)

        ####################################### settings #########################################
        label_size = 9

        ##################################### prepare data  ######################################

        # compare dbs on vs off and different dbs types for plastic and fixed shortcut
        data_df_full_sims_dict = {}
        data_df_full_sims_dict["plastic"] = dbs_on_vs_off(
            number_of_persons, shortcut=True
        )
        data_df_full_sims_dict["fixed"] = dbs_on_vs_off(
            number_of_persons, shortcut=False
        )

        # create a plot comparing dbs_states of simulations
        # for this plot combine fixed and plastic shortcut data with a new column "shortcut_type"
        data_df_full_sims_dict["plastic"]["shortcut_type"] = "plastic"
        data_df_full_sims_dict["fixed"]["shortcut_type"] = "fixed"
        data_df_full_sims_compare_dbs = pd.concat(
            [data_df_full_sims_dict["plastic"], data_df_full_sims_dict["fixed"]]
        )
        # exclude afferent
        data_df_full_sims_compare_dbs = data_df_full_sims_compare_dbs[
            data_df_full_sims_compare_dbs["dbs_state"] != "afferent"
        ]
        # order the dbs states (dbs-off, suppression, efferent, passing fibers, dbs-comb)
        data_df_full_sims_compare_dbs["dbs_state"] = pd.Categorical(
            data_df_full_sims_compare_dbs["dbs_state"],
            categories=[
                "dbs-off",
                "suppression",
                "efferent",
                "passing fibers",
                "dbs-comb",
            ],
            ordered=True,
        )

        ####################### plot fig/simulation_data_difference_dbs_on_off_100  ############################

        # 2 (plastic/fixed) times n (number of sessions) subplots
        fig_compare_dbs, axs_compare_dbs = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(16 / 2.54, 14 / 2.54),
            sharex=True,
            sharey=True,
        )

        # for loop over shortcut_type
        for shortcut_type_id, shortcut_type in enumerate(["plastic", "fixed"]):

            # boxplot with x=shortcut_type, y=unrewarded_decisions, hue=dbs_state
            sns.boxplot(
                data=data_df_full_sims_compare_dbs[
                    data_df_full_sims_compare_dbs["shortcut_type"] == shortcut_type
                ],
                x="session",
                y="unrewarded_decisions",
                hue="dbs_state",
                hue_order=[
                    "dbs-off",
                    "suppression",
                    "efferent",
                    "passing fibers",
                    "dbs-comb",
                ],
                ax=axs_compare_dbs[shortcut_type_id],
                palette={
                    "suppression": (1, 0.7, 0.7, 0.8),
                    "efferent": (1, 0.5, 0.5, 0.8),
                    "dbs-comb": (0.8, 0, 0, 0.8),
                    "dbs-off": (0, 0, 0.65),
                    "passing fibers": (1, 0.3, 0.3, 0.8),
                },
                boxprops=dict(
                    edgecolor="black", linewidth=1
                ),  # Schwarzer Rahmen der Box
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                medianprops=dict(color="black", linewidth=1),
                flierprops={
                    "marker": "o",
                    "color": "black",
                    "markersize": 4,
                    "markeredgecolor": "black",
                    "markerfacecolor": "none",
                },
                linewidth=1,
            )

            ################ axis settings #####################

            axs_compare_dbs[shortcut_type_id].tick_params(
                axis="both", labelsize=label_size
            )
            axs_compare_dbs[shortcut_type_id].set_ylabel(
                "unrewarded decisions", fontweight="bold", fontsize=label_size
            )
            axs_compare_dbs[shortcut_type_id].set_xlabel(
                "Session", fontweight="bold", fontsize=label_size
            )
            if shortcut_type_id == 0:
                axs_compare_dbs[shortcut_type_id].set_xticklabels([])
                axs_compare_dbs[shortcut_type_id].set_xlabel("")
                axs_compare_dbs[shortcut_type_id].get_legend().remove()

            ################# significance * #################

            # function for significance over the boxplots
            def add_star(ax, x1, x2, y):
                """Add asterisks for significant differences."""
                y_offset = 0.5
                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + y_offset, y + y_offset, y],
                    color="black",
                    linewidth=1,
                )
                ax.text((x1 + x2) / 2, y + 0.1, "*", fontsize=10, ha="center")

            if shortcut_type_id == 0:
                add_star(axs_compare_dbs[shortcut_type_id], 1.68, 1.84, 33)
                add_star(axs_compare_dbs[shortcut_type_id], 1.68, 2.16, 35)
                add_star(axs_compare_dbs[shortcut_type_id], 1.68, 2.32, 37)
            else:
                add_star(axs_compare_dbs[shortcut_type_id], 0.68, 0.84, 27)
                add_star(axs_compare_dbs[shortcut_type_id], 0.68, 1.16, 29)
                add_star(axs_compare_dbs[shortcut_type_id], 0.68, 1.32, 31)
                add_star(axs_compare_dbs[shortcut_type_id], 1.68, 2.32, 16)

        ############## legend #################

        handles, labels = axs_compare_dbs[0].get_legend_handles_labels()

        label_mapping = {
            "dbs-off": "DBS OFF",
            "dbs-comb": "combined",
        }
        labels = [label_mapping.get(label, label) for label in labels]

        for ax in axs_compare_dbs:
            ax.legend().remove()

        fig = plt.gcf()
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            fontsize=label_size,
            bbox_to_anchor=(0.5, -0.01),
        )

        # get the coordinates of the legend box
        legend_bbox = legend.get_window_extent()
        legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())

        ############### layout ################

        plt.tight_layout(
            pad=0,
            h_pad=1.08,
            w_pad=1.08,
            rect=[0, legend_bbox.y1 + 0.01, 1, 1],
        )

        # plot labels
        plt.text(0.005, 0.975, "A", transform=plt.gcf().transFigure, fontsize=11)
        plt.text(0.005, 0.535, "B", transform=plt.gcf().transFigure, fontsize=11)

        ############# save fig ################
        fig.savefig(
            f"fig/simulation_data_difference_dbs_on_off_{number_of_persons}.png"
        )
        plt.close(fig)

        #####################################################################################################
        #################################### !!!! FIG patient vs sims !!!!! ######################################
        #####################################################################################################

        ################################# prepare data ###########################################

        # also perform the analysis for the patient data and compare patients with sims
        data_df_full_patients = dbs_on_vs_off(number_of_persons=None)

        # change the values of the column subject to "patient X" and "simulation X"
        # where X is the previous value of the subject column
        data_df_full_patients["subject"] = "patients " + data_df_full_patients[
            "subject"
        ].astype(str)
        data_df_full_sims_dict["plastic"]["subject"] = (
            "simulations " + data_df_full_sims_dict["plastic"]["subject"].astype(str)
        )
        data_df_full_sims_dict["fixed"]["subject"] = (
            "simulations " + data_df_full_sims_dict["fixed"]["subject"].astype(str)
        )

        # create a figure with m (number of sessions) times 2 (plastic/fixed)
        # subplots
        fig_fixed_and_plastic, axs_fixed_and_plastic = plt.subplots(
            nrows=1,
            ncols=len(data_df_full_sims_dict["plastic"]["session"].unique()) + 1,
            figsize=(19 / 2.54, 7 / 2.54),
            sharex=True,
            sharey=True,
        )

        # loop over plastic and fixed shortcut
        for shortcut_type_id, shortcut_type in enumerate(["plastic", "fixed"]):

            # create a figure with n (number dbs_states - 2) time m (number of sessions)
            # subplots
            fig_full, axs_full = plt.subplots(
                nrows=len(data_df_full_sims_dict["plastic"]["dbs_state"].unique()) - 2,
                ncols=len(data_df_full_sims_dict["plastic"]["session"].unique()),
                figsize=(14 / 2.54, 18 / 2.54),
                sharex=True,
                sharey=True,
            )

            data_df_full_sims = data_df_full_sims_dict[shortcut_type]
            # for loop over dbs states
            dbs_state_id = 0
            for dbs_state in data_df_full_sims["dbs_state"].unique():
                # skip afferent and dbs-off
                if dbs_state in ["afferent", "dbs-off"]:
                    continue
                # create a figure with m (number of sessions) subplots
                fig, axs = plt.subplots(
                    nrows=1,
                    ncols=len(data_df_full_sims_dict["plastic"]["session"].unique()),
                    figsize=(14 / 2.54, 7 / 2.54),
                    sharex=True,
                    sharey=True,
                )
                # for loop over sessions
                for session_id, session in enumerate(
                    data_df_full_sims["session"].unique()
                ):
                    # filter the simulation data for the current dbs state and dbs-off
                    data_df_full_sims_filtered = data_df_full_sims[
                        data_df_full_sims["dbs_state"].isin(["dbs-off", dbs_state])
                    ].copy()
                    # rename the dbs state of the sims to dbs-on
                    data_df_full_sims_filtered.loc[
                        data_df_full_sims_filtered["dbs_state"] == dbs_state,
                        "dbs_state",
                    ] = "dbs-on"
                    # combine the patient and simulation data and add the column "subject_type"
                    data_df_full_sims_filtered["subject_type"] = "simulations"
                    data_df_full_patients["subject_type"] = "patients"
                    data_df_full_combined = pd.concat(
                        [data_df_full_sims_filtered, data_df_full_patients]
                    ).copy()
                    # filter data to only contain session
                    data_df_full_combined = data_df_full_combined[
                        data_df_full_combined["session"] == session
                    ]
                    # order the subject_type (patient, simulation)
                    data_df_full_combined["subject_type"] = pd.Categorical(
                        data_df_full_combined["subject_type"],
                        categories=["patients", "simulations"],
                        ordered=True,
                    )
                    # perform a 2 way repeated measures ANOVA with between factor
                    # "subject_type" and within factor "dbs_state"
                    aov = pg.mixed_anova(
                        data=data_df_full_combined,
                        dv="unrewarded_decisions",
                        within="dbs_state",
                        subject="subject",
                        between="subject_type",
                    )
                    # save results
                    with open(
                        f"statistic/patients_vs_sims_{number_of_persons}_{shortcut_type}_{dbs_state}_ses_{session}.txt",
                        "w",
                    ) as fh:
                        fh.write(aov.round(3).to_string())

                    # perform post hoc t-tests
                    result = pg.pairwise_tests(
                        data=data_df_full_combined,
                        dv="unrewarded_decisions",
                        within="dbs_state",
                        subject="subject",
                        between="subject_type",
                        padjust="bonf",
                        return_desc=True,
                    )
                    # save results in the same file
                    with open(
                        f"statistic/patients_vs_sims_{number_of_persons}_{shortcut_type}_{dbs_state}_ses_{session}.txt",
                        "a",
                    ) as fh:
                        fh.write("\n\npost hoc t-tests:\n")
                        fh.write(result.round(3).to_string())

                    ################################## plot settings #########################################

                    label_size = 9
                    dbs_red = (0.8, 0, 0, 0.8)
                    dbs_blue = (0, 0, 0.65)

                    ################################ 3x1 figure ######################################

                    # create boxplots with seaborn with x=subjec_type, y=unrewarded_decisions, hue=dbs_state
                    sns.boxplot(
                        data=data_df_full_combined,
                        x="subject_type",
                        y="unrewarded_decisions",
                        hue="dbs_state",
                        ax=axs[session_id],
                        palette={"dbs-off": dbs_blue, "dbs-on": dbs_red},
                        boxprops=dict(edgecolor="black", linewidth=1),
                        whiskerprops=dict(color="black", linewidth=1),
                        capprops=dict(color="black", linewidth=1),
                        medianprops=dict(color="black", linewidth=1),
                        flierprops={
                            "marker": "o",
                            "color": "black",
                            "markersize": 4,
                            "markeredgecolor": "black",
                            "markerfacecolor": "none",
                        },
                        linewidth=1,
                    )

                    ################ axis settings #####################

                    axs[session_id].tick_params(axis="both", labelsize=label_size)
                    axs[session_id].set_ylabel(
                        "unrewarded decisions", fontweight="bold", fontsize=label_size
                    )
                    axs[session_id].set_xlabel(
                        "subject type", fontweight="bold", fontsize=label_size
                    )
                    # axs[session_id].set_xticklabels(["patients", "simulations"])

                    if session_id > 0:
                        axs[session_id].set_ylabel("")

                    for column_idx, column_label in enumerate(
                        ["Session 1", "Session 2", "Session 3"]
                    ):
                        axs[column_idx].set_title(
                            column_label, fontweight="bold", fontsize=label_size
                        )

                    ############## legend #################

                    handles, labels = axs[0].get_legend_handles_labels()

                    label_mapping = {
                        "dbs-off": "DBS OFF",
                        "dbs-on": "DBS ON",
                    }
                    labels = [label_mapping.get(label, label) for label in labels]

                    axs[session_id].legend().remove()

                    # fig = plt.gcf()
                    legend = fig.legend(
                        handles,
                        labels,
                        loc="lower center",
                        ncol=2,
                        fontsize=label_size,
                        bbox_to_anchor=(0.5, -0.01),
                    )

                    # get the coordinates of the legend box
                    legend_bbox = legend.get_window_extent()
                    legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())

                    ################################### 3x4 figure ######################################

                    sns.boxplot(
                        data=data_df_full_combined,
                        x="subject_type",
                        y="unrewarded_decisions",
                        hue="dbs_state",
                        ax=axs_full[dbs_state_id, session_id],
                        palette={"dbs-off": dbs_blue, "dbs-on": dbs_red},
                        boxprops=dict(edgecolor="black", linewidth=1),
                        whiskerprops=dict(color="black", linewidth=1),
                        capprops=dict(color="black", linewidth=1),
                        medianprops=dict(color="black", linewidth=1),
                        flierprops={
                            "marker": "o",
                            "color": "black",
                            "markersize": 4,
                            "markeredgecolor": "black",
                            "markerfacecolor": "none",
                        },
                        linewidth=1,
                    )

                    ################ axis settings #####################

                    axs_full[dbs_state_id, session_id].tick_params(
                        axis="both", labelsize=label_size
                    )
                    axs_full[dbs_state_id, session_id].set_ylabel(
                        "unrewarded decisions", fontweight="bold", fontsize=label_size
                    )
                    axs_full[dbs_state_id, session_id].set_xlabel(
                        "subject type", fontweight="bold", fontsize=label_size
                    )

                    if session_id >= 1:
                        axs_full[dbs_state_id, session_id].set_ylabel("")
                    else:
                        axs_full[dbs_state_id, session_id].tick_params(
                            axis="y", labelsize=label_size
                        )

                    if dbs_state_id < 3:
                        axs_full[dbs_state_id, session_id].set_xticklabels([])
                        axs_full[dbs_state_id, session_id].set_xlabel("")

                    for column_idx, column_label in enumerate(
                        ["Session 1", "Session 2", "Session 3"]
                    ):
                        axs_full[0, column_idx].set_title(
                            column_label, fontweight="bold", fontsize=label_size
                        )

                    ############## legend #################

                    handles_full, labels_full = axs_full[1][
                        1
                    ].get_legend_handles_labels()

                    label_mapping_full = {
                        "dbs-off": "DBS OFF",
                        "dbs-on": "DBS ON",
                    }
                    labels_full = [
                        label_mapping_full.get(label, label) for label in labels_full
                    ]

                    axs_full[dbs_state_id, session_id].legend().remove()

                    # fig1 = plt.gcf()
                    legend_full = fig_full.legend(
                        handles_full,
                        labels_full,
                        loc="lower center",
                        ncol=2,
                        fontsize=label_size,
                        bbox_to_anchor=(0.5, -0.01),
                    )

                    # get the coordinates of the legend box
                    legend_bbox_full = legend_full.get_window_extent()
                    legend_bbox_full = legend_bbox_full.transformed(
                        fig_full.transFigure.inverted()
                    )

                    ################# significance * #################

                    # function for significance over the boxplots
                    def add_star(ax, x1, x2, y):
                        """Add asterisks for significant differences."""
                        y_offset = 0.5
                        ax.plot(
                            [x1, x1, x2, x2],
                            [y, y + y_offset, y + y_offset, y],
                            color="black",
                            linewidth=1,
                        )
                        ax.text((x1 + x2) / 2, y + 0.1, "*", fontsize=10, ha="center")

                    if shortcut_type_id == 0:
                        if session_id == 1:
                            if dbs_state_id == 0:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 24
                                )
                            if dbs_state_id == 1:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 28
                                )
                            if dbs_state_id == 2:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 28
                                )
                            if dbs_state_id == 3:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 24
                                )

                    else:
                        if session_id == 1:
                            if dbs_state_id == 0:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 26
                                )
                            if dbs_state_id == 2:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 26
                                )
                            if dbs_state_id == 3:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 25
                                )
                        if session_id == 2:
                            if dbs_state_id == 0:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 25
                                )
                                add_star(
                                    axs_full[dbs_state_id, session_id], -0.2, 0.8, 28
                                )
                            if dbs_state_id == 1:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 25
                                )
                                add_star(
                                    axs_full[dbs_state_id, session_id], -0.2, 0.8, 28
                                )
                            if dbs_state_id == 2:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 25
                                )
                                add_star(
                                    axs_full[dbs_state_id, session_id], -0.2, 0.8, 28
                                )
                            if dbs_state_id == 3:
                                add_star(
                                    axs_full[dbs_state_id, session_id], 0.20, 1.20, 25
                                )
                                add_star(
                                    axs_full[dbs_state_id, session_id], -0.2, 0.8, 28
                                )

                    ################################ 4x1 figure ######################################
                    # first three pics
                    if dbs_state == "dbs-comb" and shortcut_type == "plastic":
                        sns.boxplot(
                            data=data_df_full_combined,
                            x="subject_type",
                            y="unrewarded_decisions",
                            hue="dbs_state",
                            ax=axs_fixed_and_plastic[session_id],
                            palette={"dbs-off": dbs_blue, "dbs-on": dbs_red},
                            boxprops=dict(edgecolor="black", linewidth=1),
                            whiskerprops=dict(color="black", linewidth=1),
                            capprops=dict(color="black", linewidth=1),
                            medianprops=dict(color="black", linewidth=1),
                            flierprops={
                                "marker": "o",
                                "color": "black",
                                "markersize": 4,
                                "markeredgecolor": "black",
                                "markerfacecolor": "none",
                            },
                            linewidth=1,
                        )
                        axs_fixed_and_plastic[session_id].get_legend().remove()

                        ################ axis settings #####################

                        axs_fixed_and_plastic[session_id].tick_params(
                            axis="both", labelsize=label_size
                        )
                        axs_fixed_and_plastic[session_id].set_ylabel(
                            "unrewarded decisions",
                            fontweight="bold",
                            fontsize=label_size,
                        )
                        axs_fixed_and_plastic[session_id].set_xlabel(
                            "subject type", fontweight="bold", fontsize=label_size
                        )

                        if session_id > 0:
                            axs_fixed_and_plastic[session_id].set_ylabel("")

                        ################# significance * #################

                        # function for significance over the boxplots
                        def add_star(ax, x1, x2, y):

                            y_offset = 0.5
                            ax.plot(
                                [x1, x1, x2, x2],
                                [y, y + y_offset, y + y_offset, y],
                                color="black",
                                linewidth=1.0,
                            )
                            ax.text(
                                (x1 + x2) / 2, y + 0.1, "*", fontsize=10, ha="center"
                            )

                        if session_id == 1:
                            add_star(axs_fixed_and_plastic[session_id], 0.20, 1.20, 24)

                    elif (
                        dbs_state == "dbs-comb"
                        and shortcut_type == "fixed"
                        and session_id == 2
                    ):
                        sns.boxplot(
                            # only data for subject_type == "simulation"
                            data=data_df_full_combined,
                            x="subject_type",
                            y="unrewarded_decisions",
                            hue="dbs_state",
                            ax=axs_fixed_and_plastic[-1],
                            palette={"dbs-off": dbs_blue, "dbs-on": dbs_red},
                            boxprops=dict(edgecolor="black", linewidth=1),
                            whiskerprops=dict(color="black", linewidth=1),
                            capprops=dict(color="black", linewidth=1),
                            medianprops=dict(color="black", linewidth=1),
                            flierprops={
                                "marker": "o",
                                "color": "black",
                                "markersize": 4,
                                "markeredgecolor": "black",
                                "markerfacecolor": "none",
                            },
                            linewidth=1,
                        )
                        axs_fixed_and_plastic[-1].get_legend().remove()
                        axs_fixed_and_plastic[-1].set_ylabel("")
                        axs_fixed_and_plastic[-1].tick_params(
                            axis="both", labelsize=label_size
                        )
                        axs_fixed_and_plastic[-1].set_xlabel(
                            "subject type", fontweight="bold", fontsize=label_size
                        )

                        ################# significance * #################

                        # function for significance over the boxplots
                        def add_star(ax, x1, x2, y):

                            y_offset = 0.5
                            ax.plot(
                                [x1, x1, x2, x2],
                                [y, y + y_offset, y + y_offset, y],
                                color="black",
                                linewidth=1.0,
                            )
                            ax.text(
                                (x1 + x2) / 2, y + 0.1, "*", fontsize=10, ha="center"
                            )

                        add_star(axs_fixed_and_plastic[-1], 0.20, 1.20, 24)
                        add_star(axs_fixed_and_plastic[-1], -0.20, 0.80, 27)

                    ###################### titels #######################

                    for column_idx, column_label in enumerate(
                        [
                            "Session 1",
                            "Session 2",
                            "Session 3",
                            "Session 3\n(fixed shortcut)",
                        ]
                    ):
                        axs_fixed_and_plastic[column_idx].set_title(
                            column_label, fontweight="bold", fontsize=label_size
                        )

                    ############## legend #################

                    handles_fixed_and_plastic, labels_fixed_and_plastic = (
                        axs_fixed_and_plastic[0].get_legend_handles_labels()
                    )

                    label_mapping_fixed_and_plastic = {
                        "dbs-off": "DBS OFF",
                        "dbs-on": "DBS ON",
                    }
                    labels_fixed_and_plastic = [
                        label_mapping_fixed_and_plastic.get(label, label)
                        for label in labels_fixed_and_plastic
                    ]

                    # fig = plt.gcf()
                    legend_fixed_and_plastic = fig_fixed_and_plastic.legend(
                        handles_fixed_and_plastic,
                        labels_fixed_and_plastic,
                        loc="lower center",
                        ncol=2,
                        fontsize=label_size,
                        bbox_to_anchor=(0.5, -0.01),
                    )

                    # get the coordinates of the legend box
                    legend_bbox_fixed_and_plastic = (
                        legend_fixed_and_plastic.get_window_extent()
                    )
                    legend_bbox_fixed_and_plastic = (
                        legend_bbox_fixed_and_plastic.transformed(
                            fig_fixed_and_plastic.transFigure.inverted()
                        )
                    )

                ############################### save figs, layout & labels #########################################

                #################### 3x1 figure ######################
                fig.set_linewidth(1)

                fig.tight_layout(
                    pad=0,
                    h_pad=1.08,
                    w_pad=1.08,
                    rect=[0, legend_bbox.y1 + 0.01, 1 + 0.002, 1],
                )
                fig.savefig(
                    f"fig/patients_vs_sims_{number_of_persons}_{shortcut_type}_{dbs_state}.png",
                )
                plt.close(fig)
                dbs_state_id += 1

            ######################### 3x4 figure #######################

            if shortcut_type_id == 0:
                fig_full.tight_layout(
                    pad=0,
                    h_pad=1.08,
                    w_pad=1.08,
                    rect=[0.03, legend_bbox_full.y1 + 0.01, 1, 1],
                )
            else:
                fig_full.tight_layout(
                    pad=0,
                    h_pad=1.08,
                    w_pad=1.08,
                    rect=[0.03, legend_bbox_full.y1 + 0.01, 1, 1],
                )

            # plot labels
            plt.text(0.001, 0.965, "A", transform=plt.gcf().transFigure, fontsize=11)
            plt.text(0.001, 0.74, "B", transform=plt.gcf().transFigure, fontsize=11)
            plt.text(0.001, 0.512, "C", transform=plt.gcf().transFigure, fontsize=11)
            plt.text(0.001, 0.29, "D", transform=plt.gcf().transFigure, fontsize=11)

            fig_full.savefig(
                f"fig/patients_vs_sims_{number_of_persons}_{shortcut_type}.png",
            )
            plt.close(fig_full)

        ############################# 4x1 figure #######################

        fig_fixed_and_plastic.tight_layout(
            pad=0,
            h_pad=1.08,
            w_pad=1.08,
            rect=[0, legend_bbox_fixed_and_plastic.y1 + 0.01, 1 - 0.006, 1],
        )
        fig_fixed_and_plastic.savefig(
            f"fig/patients_vs_sims_plastic_and_fixed_{number_of_persons}.png",
        )
        plt.close(fig_fixed_and_plastic)

        #####################################################################################################
        #################################### !!!! FIG DBS ON OFF !!!!! ######################################
        #####################################################################################################

    if H3:
        dbs_state = [
            "DBS-OFF vs. suppression:",
            "DBS-OFF vs. EFFERENT:",
            "DBS-OFF vs. AFFERENT:",
            "DBS-OFF vs. PASSING FIBERS:",
            "DBS-OFF vs. DBS_ALL:",
        ]

        table_H3 = {
            "DBS": [
                "dbs-off vs. suppression",
                "dbs-off vs. efferent",
                "dbs-off vs. afferent",
                "dbs-off vs. passing fibers",
                "dbs-off vs. dbs-comb",
            ],
            "S1 t": [],
            "S1 p": [],
            "S1 df": [],
            "S1 diff": [],
            "S1 norm": [],
            "S1 var": [],
            "S2 t": [],
            "S2 p": [],
            "S2 df": [],
            "S2 diff": [],
            "S2 norm": [],
            "S2 var": [],
            "S3 t": [],
            "S3 p": [],
            "S3 df": [],
            "S3 diff": [],
            "S3 norm": [],
            "S3 var": [],
        }

        for i in range(5):
            ########################## load data #############################

            DataBA_1 = read_json_data(
                f"data/simulation_data/Results_Shortcut1_DBS_State0.json"
            )
            DataBA_1 = processing_habit_data(DataBA_1, number_of_persons)

            DataBA_2 = read_json_data(
                f"data/simulation_data/Results_Shortcut1_DBS_State{i+1}.json"
            )
            DataBA_2 = processing_habit_data(DataBA_2, number_of_persons)

            ########################## T-Test ###################################
            print("#############################################")
            print(dbs_state[i], "\n")
            # Session1
            Data1 = DataBA_1[:, 0]
            Data2 = DataBA_2[:, 0]
            print("Session 1:")
            ttest = ttest_data(Data1, Data2)
            table_H3["S1 t"].append(ttest[0])
            table_H3["S1 p"].append(ttest[1])
            table_H3["S1 df"].append(ttest[2])
            table_H3["S1 diff"].append(ttest[3])
            table_H3["S1 norm"].append(ttest[4])
            table_H3["S1 var"].append(ttest[5])

            # Session2
            Data3 = DataBA_1[:, 1]
            Data4 = DataBA_2[:, 1]
            print("Session 2:")
            ttest = ttest_data(Data3, Data4)
            table_H3["S2 t"].append(ttest[0])
            table_H3["S2 p"].append(ttest[1])
            table_H3["S2 df"].append(ttest[2])
            table_H3["S2 diff"].append(ttest[3])
            table_H3["S2 norm"].append(ttest[4])
            table_H3["S2 var"].append(ttest[5])

            # Session3
            Data5 = DataBA_1[:, 2]
            Data6 = DataBA_2[:, 2]
            print("Session 3:")
            ttest = ttest_data(Data5, Data6)
            table_H3["S3 t"].append(ttest[0])
            table_H3["S3 p"].append(ttest[1])
            table_H3["S3 df"].append(ttest[2])
            table_H3["S3 diff"].append(ttest[3])
            table_H3["S3 norm"].append(ttest[4])
            table_H3["S3 var"].append(ttest[5])

        save_table(
            table_H3,
            f"statistic/simulation_data_difference_dbs_on_off_{number_of_persons}",
        )


#####################################################################################################
################################## load simulation data - ANOVA #####################################
#####################################################################################################


# 2x2 ANOVA
def anova_load_simulation(number_of_persons):

    dbs_state = [
        "dbs-off",
        "suppression",
        "efferent",
        "afferent",
        "passing-fibres",
        "dbs-comb",
    ]

    for i in range(1, 6):
        ########################## load data #############################

        # skip afferent
        if i == 3:
            continue

        # load and process data
        Data1 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_2.json"
        )
        Data1 = processing_habit_session3(Data1, number_of_persons)
        Data2 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_3.json"
        )
        Data2 = processing_habit_session3(Data2, number_of_persons)
        Data3 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_4.json"
        )
        Data3 = processing_habit_session3(Data3, number_of_persons)
        Data4 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_5.json"
        )
        Data4 = processing_habit_session3(Data4, number_of_persons)

        ########################## ANOVA ###################################

        anova = rm_anova_group(Data1, Data2, Data3, Data4, i)

        save_table(
            anova,
            f"statistic/load_simulation_anova_dbs_state_{dbs_state[i]}_N{number_of_persons}",
        )


# pairwise T-Test
def anova_load_simulation_ttest(number_of_persons):

    dbs_state = [
        "dbs-off",
        "suppression",
        "efferent",
        "afferent",
        "passing-fibres",
        "dbs-comb",
    ]

    for i in range(1, 6):
        ########################## load data #############################

        # skip afferent
        if i == 3:
            continue

        # load and process data
        Data1 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_2.json"
        )
        Data1 = processing_habit_session3(Data1, number_of_persons)
        Data2 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_3.json"
        )
        Data2 = processing_habit_session3(Data2, number_of_persons)
        Data3 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_4.json"
        )
        Data3 = processing_habit_session3(Data3, number_of_persons)
        Data4 = read_json_data(
            f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_5.json"
        )
        Data4 = processing_habit_session3(Data4, number_of_persons)

        ########################## ANOVA ###################################

        ttest = pairwise_ttest(Data1, Data2, Data3, Data4, i)

        save_table(
            ttest,
            f"statistic/load_simulation_pairwise_ttest_dbs_state_{dbs_state[i]}_N{number_of_persons}",
        )


#####################################################################################################
################################### previously selected effect ######################################
#####################################################################################################


def get_subject_choice_trial_variables(data_df: pd.DataFrame, choice_list: list):
    """
    Calculate the previously selected, previously rewarded and selected variables for
    each choice and trial.

    Args:
        data_df (pd.DataFrame):
            DataFrame containing the subject, trial, choice and reward columns
        choice_list (list):
            List of choices for which the variables should be calculated

    Returns:
        pd.DataFrame:
            DataFrame containing the subject, trial, choice, previously_selected,
            previously_rewarded, and selected columns
    """
    ret = {
        "choice": [],
        "subject": [],
        "trial": [],
        "previously_selected": [],
        "previously_rewarded": [],
        "selected": [],
    }

    choices_uniq_arr = data_df["choice"].unique()
    # Set the time window for the previously selected and previously rewarded variables
    # reward_time needs to be smaller or equal to selected_time
    reward_time = 5

    # Precompute subject indices
    subject_indices = {
        subject: data_df[data_df["subject"] == subject].index
        for subject in data_df["subject"].unique()
    }

    for choice in choice_list:
        for subject, indices in subject_indices.items():
            trials = data_df.loc[indices, "trial"].values
            choices = data_df.loc[indices, "choice"].values
            rewards = data_df.loc[indices, "reward"].values

            for i in trials[len(trials) // 2 :]:
                if i < reward_time:
                    continue

                ret["choice"].append(choice)
                ret["subject"].append(subject)
                ret["trial"].append(i)

                # Calculate previously selected
                selected_window = choices[:i]
                ret["previously_selected"].append(
                    round(np.sum(selected_window == choice) / len(selected_window), 3)
                )

                # Calculate previously rewarded
                reward_window = rewards[i - reward_time : i]

                choice_unique_previous_rewards = np.array(
                    [
                        np.sum(
                            reward_window[choices[i - reward_time : i] == uniq_choice]
                        )
                        for uniq_choice in choices_uniq_arr
                    ]
                )

                if (
                    choice_unique_previous_rewards[choices_uniq_arr == choice]
                    > choice_unique_previous_rewards[choices_uniq_arr != choice]
                ).all():
                    # The choice is the most rewarded one
                    ret["previously_rewarded"].append(1)
                elif (
                    choice_unique_previous_rewards[choices_uniq_arr == choice]
                    >= choice_unique_previous_rewards[choices_uniq_arr != choice]
                ).all():
                    # The choice is the most rewarded one together with other choices
                    ret["previously_rewarded"].append(0.5)
                else:
                    # Other choices are more rewarded
                    ret["previously_rewarded"].append(0)

                # Set selected
                ret["selected"].append(int(choices[i] == choice))

    return pd.DataFrame(ret)


def partial_corr_x_y_cv(variables_df: pd.DataFrame):
    """
    Calculate the partial correlation between previously selected and selected
    while controlling for previously rewarded and vice versa.

    Args:
        variables_df (pd.DataFrame):
            DataFrame containing the subject, trial, choice, previously_selected,
            previously_rewarded, and selected columns

    Returns:
        results_txt (str):
            String containing the results of the partial correlation analysis
        figures_dict (dict):
            Dictionary containing the figures generated during the analysis
    """
    results_txt = ""
    figures_dict = {}

    # Test for normality
    results_normality = pg.multivariate_normality(
        variables_df[["previously_selected", "previously_rewarded", "selected"]]
    )
    results_txt += f"Normality test:\n{results_normality}\n\n"

    # Visual inspection of the data
    for variable in ["previously_selected", "previously_rewarded"]:
        fig, ax = plt.subplots()
        # Create a pivot table to aggregate counts
        heatmap_data = variables_df.pivot_table(
            index="selected",
            columns=variable,
            aggfunc="size",
            fill_value=0,
        )
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="viridis", ax=ax)
        ax.set_title(f"Counts of {variable} by selected")
        figures_dict[f"heatmap_{variable}"] = fig

    # Partial correlation analysis
    cor_overview = pg.pairwise_corr(
        variables_df,
        columns=["previously_selected", "previously_rewarded", "selected"],
        method="spearman",
    ).round(3)
    results_txt += f"Pairwise correlations (not partial!):\n{cor_overview}\n\n"
    par_cor_selected = pg.partial_corr(
        data=variables_df,
        x="previously_selected",
        y="selected",
        covar="previously_rewarded",
        method="spearman",
    ).round(3)
    par_cor_rewarded = pg.partial_corr(
        data=variables_df,
        x="previously_rewarded",
        y="selected",
        covar="previously_selected",
        method="spearman",
    ).round(3)
    results_txt += (
        f"Partial correlation previously selected and selected:\n{par_cor_selected}\n\n"
    )
    results_txt += (
        f"Partial correlation previously rewarded and selected:\n{par_cor_rewarded}\n\n"
    )

    return results_txt, figures_dict


def load_data_previously_selected(
    subject_type: str,
    shortcut_type: str,
    dbs_state: str,
    dbs_variant: str,
    switch_choice_manipulation: float | None,
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
            "suppression", "efferent" or "dbs-comb"
        switch_choice_manipulation (float | None):
            Fraction of trials for which the choice should be switched.

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

    elif subject_type == "simulation":
        shortcut_load = {"plastic": 1, "fixed": 0}[shortcut_type]
        if dbs_state == "OFF":
            dbs_load = 0
        elif dbs_state == "ON":
            dbs_load = {"suppression": 1, "efferent": 2, "combined": 5}[dbs_variant]
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

    # swtich choice randomly for a fraction of the trials defined by
    # switch_choice_manipulation
    if switch_choice_manipulation != None:
        rng = np.random.default_rng(42)
        choices = np.array(ret["choice"])
        choices_uniq = np.unique(ret["choice"])
        # only a fraction of the choices should be switched
        switch_mask = rng.choice(
            [0, 1],
            len(choices),
            p=[1 - switch_choice_manipulation, switch_choice_manipulation],
        ).astype(bool)
        choices_copy = choices.copy()
        choices[switch_mask & (choices_copy == choices_uniq[0])] = choices_uniq[1]
        choices[switch_mask & (choices_copy == choices_uniq[1])] = choices_uniq[0]
        ret["choice"] = choices.tolist()

    return pd.DataFrame(ret)


def group_wide_corr_analysis(data_df, choice_list):
    """
    Perform correlation analysis for the whole group.

    Args:
        data_df (pd.DataFrame):
            DataFrame containing the subject, trial, choice and reward columns
        choice_list (list):
            List of choices for which the correlation analysis should be done

    Returns:
        results_txt (str):
            String containing the results of the correlation analysis
        figures_dict (dict):
            Dictionary containing the figures generated during the analysis
    """
    # obtain the:
    # - first independent variable by calculating for both possible choices
    #   for each trial how often they were selected previously
    # - second independent variable by calculating for both possible choices
    #   for each trial if they are currently the most rewarded choice
    # - dependent variable for both possible choices for each trial if they were
    #   selected in the current trial
    # all together in one dataframe
    subject_choice_trial_variables_df = get_subject_choice_trial_variables(
        data_df=data_df, choice_list=choice_list
    )

    # calculate partial correlation analyses using the variables of the dataframe
    return partial_corr_x_y_cv(variables_df=subject_choice_trial_variables_df)


def previously_selected():
    """
    Perform the correlation analysis for the previously selected and previously rewarded
    variables. Therefore check if choices are correlated with the previous selections
    also controlling for the previous rewards.
    """
    # for which choices the correlation analysis should be done
    choice_list = [1]
    # set for which groups correlation analyses should be done
    subject_type_list = ["patient", "simulation"]
    shortcut_type_list = ["plastic", "fixed"]
    dbs_variant_list = ["suppression", "efferent", "dbs-comb"]
    dbs_state_list = ["ON", "OFF"]
    switch_choice_manipulation_list = [None, 0.05, 0.1]
    # get all combinations to get groups
    groups = list(
        itertools.product(
            subject_type_list,
            shortcut_type_list,
            dbs_variant_list,
            dbs_state_list,
            switch_choice_manipulation_list,
        )
    )
    # remove some groups
    actual_groups = []
    for (
        subject_type,
        shortcut_type,
        dbs_variant,
        dbs_state,
        switch_choice_manipulation,
    ) in groups:
        # for patients skip all variations of shortcut_type, dbs_variant and switch_choice_manipulation
        if subject_type == "patient" and not (
            shortcut_type == shortcut_type_list[0]
            and dbs_variant == dbs_variant_list[0]
            and switch_choice_manipulation == None
        ):
            continue
        # do the switch choice manipulation only for the plastic shortcut and dbs OFF
        if switch_choice_manipulation != None and not (
            shortcut_type == "plastic"
            and dbs_variant == dbs_variant_list[0]
            and dbs_state == "OFF"
        ):
            continue
        actual_groups.append(
            (
                subject_type,
                shortcut_type,
                dbs_variant,
                dbs_state,
                switch_choice_manipulation,
            )
        )

    # loop over groups
    for (
        subject_type,
        shortcut_type,
        dbs_variant,
        dbs_state,
        switch_choice_manipulation,
    ) in tqdm(actual_groups):

        # load the data of the subjects of the group
        data_df = load_data_previously_selected(
            subject_type=subject_type,
            shortcut_type=shortcut_type,
            dbs_state=dbs_state,
            dbs_variant=dbs_variant,
            switch_choice_manipulation=switch_choice_manipulation,
        )

        # perform correlation anaylsis for the whole group
        results_txt, figures_dict = group_wide_corr_analysis(
            data_df, choice_list=choice_list
        )

        # save the results and figures
        with open(
            f"statistic/group_wide_corr_analysis_{subject_type}_{shortcut_type}_{dbs_variant}_{dbs_state}_{switch_choice_manipulation}.txt",
            "w",
        ) as f:
            f.write(results_txt)

        for fig_name, fig in figures_dict.items():
            fig.savefig(
                f"statistic/group_wide_corr_analysis_{fig_name}_{subject_type}_{shortcut_type}_{dbs_variant}_{dbs_state}_{switch_choice_manipulation}.png"
            )
            plt.close(fig)

        # get the subject-wise correlations
        # correlations = get_subject_wise_correlations(data_df)


#####################################################################################################
################################### MANOVA change activity ##########################################
#####################################################################################################


def manova_change_activity():

    for session in range(1, 2):
        # initial
        number_of_simulations = 100
        data_dbs1 = []
        data_dbs2 = []
        data_dbs3 = []
        data_dbs4 = []
        data_dbs5 = []
        data_dbs6 = []

        ################################################# load data #################################################
        filepath1 = f"data/activity_change/activity_change_dbs_state0_session{session}"
        filepath2 = f"data/activity_change/activity_change_dbs_state1_session{session}"
        filepath3 = f"data/activity_change/activity_change_dbs_state2_session{session}"
        filepath4 = f"data/activity_change/activity_change_dbs_state3_session{session}"
        filepath5 = f"data/activity_change/activity_change_dbs_state4_session{session}"
        filepath6 = f"data/activity_change/activity_change_dbs_state5_session{session}"

        for i in range(number_of_simulations):
            data_dbs1_load = read_json_data(filepath1 + f"_id{i}.json")
            data_dbs2_load = read_json_data(filepath2 + f"_id{i}.json")
            data_dbs3_load = read_json_data(filepath3 + f"_id{i}.json")
            data_dbs4_load = read_json_data(filepath4 + f"_id{i}.json")
            data_dbs5_load = read_json_data(filepath5 + f"_id{i}.json")
            data_dbs6_load = read_json_data(filepath6 + f"_id{i}.json")

            # append loaded data to list
            data_dbs1.append(data_dbs1_load)
            data_dbs2.append(data_dbs2_load)
            data_dbs3.append(data_dbs3_load)
            data_dbs4.append(data_dbs4_load)
            data_dbs5.append(data_dbs5_load)
            data_dbs6.append(data_dbs6_load)

        # concatenate data
        data_dbs1 = np.array(data_dbs1).T
        data_dbs2 = np.array(data_dbs2).T
        data_dbs3 = np.array(data_dbs3).T
        data_dbs4 = np.array(data_dbs4).T
        data_dbs5 = np.array(data_dbs5).T
        data_dbs6 = np.array(data_dbs6).T

        ####################################### Table mean/error #############################################

        # means
        table_mean_dbs1 = []
        table_mean_dbs2 = []
        table_mean_dbs3 = []
        table_mean_dbs4 = []
        table_mean_dbs5 = []
        table_mean_dbs6 = []

        for i in range(len(data_dbs1)):
            table_mean_dbs1.append(np.mean(data_dbs1[i]))
            table_mean_dbs2.append(np.mean(data_dbs2[i]))
            table_mean_dbs3.append(np.mean(data_dbs3[i]))
            table_mean_dbs4.append(np.mean(data_dbs4[i]))
            table_mean_dbs5.append(np.mean(data_dbs5[i]))
            table_mean_dbs6.append(np.mean(data_dbs6[i]))

        table_mean_dbsoff = [
            np.array(table_mean_dbs1[3:]),
            np.array(table_mean_dbs1[3:]),
            np.array(table_mean_dbs1[3:]),
            np.array(table_mean_dbs1[3:]),
            np.array(table_mean_dbs1[3:]),
        ]

        table_mean_dbson = [
            np.array(table_mean_dbs2[3:]),
            np.array(table_mean_dbs3[3:]),
            np.array(table_mean_dbs4[3:]),
            np.array(table_mean_dbs5[3:]),
            np.array(table_mean_dbs6[3:]),
        ]

        # Faktoren definieren
        states = ["DBS-OFF", "DBS-ON"]
        conditions = [
            "suppression",
            "efferent",
            "afferent",
            "passing-fibres",
            "dbs-comb",
        ]
        populations = ["STN", "GPi", "GPe", "Thalamus", "Cor_dec", "StrThal"]

        # Testwerte generieren
        data = []
        for state in states:
            for idx, condition in enumerate(conditions):
                if state == "DBS-OFF":
                    values = table_mean_dbsoff[idx]
                else:
                    values = table_mean_dbson[idx]
                print(values)
                data.append([state, condition] + list(values))

        # DataFrame erstellen
        columns = ["State", "Condition"] + populations
        df = pd.DataFrame(data, columns=columns)

        filename = "statistic/manova"
        save_table(df, filename)

        # Überblick über die Daten
        print("Beispieldaten:")
        print(df)

        # MANOVA durchführen
        # Zunächst werden die abhängigen Variablen (Populationsdaten) ausgewählt
        dependent_vars = "+".join(populations)

        # Formel für die MANOVA
        formula = f"{dependent_vars} ~ State * Condition"

        # MANOVA-Modell erstellen
        manova = MANOVA.from_formula(formula, data=df)

        # Ergebnisse anzeigen
        print("\nMANOVA Ergebnisse:")
        print(manova.mv_test())


def anova_change_activity():

    for session in range(1, 2):
        # initial
        number_of_simulations = 100
        data_dbs1 = []
        data_dbs2 = []
        data_dbs3 = []
        data_dbs4 = []
        data_dbs5 = []
        data_dbs6 = []

        ################################################# load data #################################################
        filepath1 = f"data/activity_change/activity_change_dbs_state0_session{session}"
        filepath2 = f"data/activity_change/activity_change_dbs_state1_session{session}"
        filepath3 = f"data/activity_change/activity_change_dbs_state2_session{session}"
        filepath4 = f"data/activity_change/activity_change_dbs_state3_session{session}"
        filepath5 = f"data/activity_change/activity_change_dbs_state4_session{session}"
        filepath6 = f"data/activity_change/activity_change_dbs_state5_session{session}"

        for i in range(number_of_simulations):
            data_dbs1_load = read_json_data(filepath1 + f"_id{i}.json")
            data_dbs2_load = read_json_data(filepath2 + f"_id{i}.json")
            data_dbs3_load = read_json_data(filepath3 + f"_id{i}.json")
            data_dbs4_load = read_json_data(filepath4 + f"_id{i}.json")
            data_dbs5_load = read_json_data(filepath5 + f"_id{i}.json")
            data_dbs6_load = read_json_data(filepath6 + f"_id{i}.json")

            # append loaded data to list
            data_dbs1.append(data_dbs1_load)
            data_dbs2.append(data_dbs2_load)
            data_dbs3.append(data_dbs3_load)
            data_dbs4.append(data_dbs4_load)
            data_dbs5.append(data_dbs5_load)
            data_dbs6.append(data_dbs6_load)

        # concatenate data
        data_dbs1 = np.array(data_dbs1).T
        data_dbs2 = np.array(data_dbs2).T
        data_dbs3 = np.array(data_dbs3).T
        data_dbs4 = np.array(data_dbs4).T
        data_dbs5 = np.array(data_dbs5).T
        data_dbs6 = np.array(data_dbs6).T

        ####################################### Table mean/error #############################################

        # means
        table_mean_dbs1 = []
        table_mean_dbs2 = []
        table_mean_dbs3 = []
        table_mean_dbs4 = []
        table_mean_dbs5 = []
        table_mean_dbs6 = []

        #        for i in range(len(data_dbs1)):
        #            table_mean_dbs1.append(np.mean(data_dbs1[i]))
        #            table_mean_dbs2.append(np.mean(data_dbs2[i]))
        #            table_mean_dbs3.append(np.mean(data_dbs3[i]))
        #            table_mean_dbs4.append(np.mean(data_dbs4[i]))
        #            table_mean_dbs5.append(np.mean(data_dbs5[i]))
        #            table_mean_dbs6.append(np.mean(data_dbs6[i]))

        for i in range(len(data_dbs1)):
            table_mean_dbs1.append(data_dbs1[i])
            table_mean_dbs2.append(data_dbs2[i])
            table_mean_dbs3.append(data_dbs3[i])
            table_mean_dbs4.append(data_dbs4[i])
            table_mean_dbs5.append(data_dbs5[i])
            table_mean_dbs6.append(data_dbs6[i])

    # table_mean_dbs1 = table_mean_dbs1[3:]
    # table_mean_dbs2 = table_mean_dbs2[3:]
    # table_mean_dbs3 = table_mean_dbs3[3:]
    # table_mean_dbs4 = table_mean_dbs4[3:]
    # table_mean_dbs5 = table_mean_dbs5[3:]
    # table_mean_dbs6 = table_mean_dbs6[3:]

    table_mean_dbs1 = np.array(table_mean_dbs1)
    table_mean_dbs2 = np.array(table_mean_dbs2)
    table_mean_dbs3 = np.array(table_mean_dbs3)
    table_mean_dbs4 = np.array(table_mean_dbs4)
    table_mean_dbs5 = np.array(table_mean_dbs5)
    table_mean_dbs6 = np.array(table_mean_dbs6)

    table_mean_dbs1 = [sublist[0] for sublist in table_mean_dbs1]
    table_mean_dbs2 = [sublist[0] for sublist in table_mean_dbs2]
    table_mean_dbs3 = [sublist[0] for sublist in table_mean_dbs3]
    table_mean_dbs4 = [sublist[0] for sublist in table_mean_dbs4]
    table_mean_dbs5 = [sublist[0] for sublist in table_mean_dbs5]
    table_mean_dbs6 = [sublist[0] for sublist in table_mean_dbs6]

    table_mean_dbs1 = np.array(table_mean_dbs1)
    table_mean_dbs2 = np.array(table_mean_dbs2)
    table_mean_dbs3 = np.array(table_mean_dbs3)
    table_mean_dbs4 = np.array(table_mean_dbs4)
    table_mean_dbs5 = np.array(table_mean_dbs5)
    table_mean_dbs6 = np.array(table_mean_dbs6)

    conditions = {
        "dbs-off": table_mean_dbs1,
        "suppression": table_mean_dbs2,
        "efferent": table_mean_dbs3,
        "afferent": table_mean_dbs4,
        "passing-fibres": table_mean_dbs5,
        "dbs-comb": table_mean_dbs6,
    }

    populations = [
        "Cor_in",
        "StrD1",
        "StrD2",
        "STN",
        "GPi",
        "GPe",
        "Thalamus",
        "Cor_dec",
        "StrThal",
    ]

    # Schritt 1: Daten für MANOVA vorbereiten
    manova_data = []
    for condition, values in conditions.items():
        manova_data.append(
            {
                "Condition": condition,
                # "State": condition,
                **dict(zip(populations, values)),
            }
        )

    df = pd.DataFrame(manova_data)

    # print(df["STN"][5])
    # print(df["STN"][4])
    filename = "statistic/manova"
    save_table(df, filename)

    """
    formula = " + ".join(populations)
    manova = MANOVA.from_formula(f"{formula} ~ Condition", data=df)
    mv_test = manova.mv_test()

    print(f"MANOVA:")
    print(mv_test)
    print()
    """

    # ANOVA für jede Variable durchführen
    anova_results = {}

    for pop in populations:
        # Durchführung der ANOVA mit Pingouin
        print(f"Ergebnis für {pop}:")

        anova = anova_group2(
            df[pop][0],
            df[pop][1],
            df[pop][2],
            df[pop][3],
            df[pop][4],
            df[pop][5],
        )
        # print(df[pop][0])
        # print(df[pop][1])
        # print(df[pop][2])
        # print(df[pop][3])
        # print(df[pop][4])
        # print(df[pop][5])

        print(anova)

    # Speichern der Ergebnisse
    # anova_results[pop] = anova[["Source", "F", "p-unc"]]

    # Ergebnisse anzeigen
    # anova_results_df = pd.concat(anova_results, axis=1)
    # print(anova_results_df)


#####################################################################################################
################################### linear regression change activity ###############################
#####################################################################################################


def linear_regression():

    ######################### data ##############################

    number_of_simulations = 100
    data_dbs1 = []
    data_dbs2 = []
    data_dbs3 = []
    data_dbs4 = []
    data_dbs5 = []
    data_dbs6 = []

    ###################### load data ###########################

    filepath1 = "data/activity_change/activity_change_dbs_state0_session1"
    filepath2 = "data/activity_change/activity_change_dbs_state1_session1"
    filepath3 = "data/activity_change/activity_change_dbs_state2_session1"
    filepath4 = "data/activity_change/activity_change_dbs_state3_session1"
    filepath5 = "data/activity_change/activity_change_dbs_state4_session1"
    filepath6 = "data/activity_change/activity_change_dbs_state5_session1"

    for i in range(number_of_simulations):
        data_dbs1_load = read_json_data(filepath1 + f"_id{i}.json")
        data_dbs2_load = read_json_data(filepath2 + f"_id{i}.json")
        data_dbs3_load = read_json_data(filepath3 + f"_id{i}.json")
        data_dbs4_load = read_json_data(filepath4 + f"_id{i}.json")
        data_dbs5_load = read_json_data(filepath5 + f"_id{i}.json")
        data_dbs6_load = read_json_data(filepath6 + f"_id{i}.json")

        # append loaded data to list
        data_dbs1.append(data_dbs1_load[0])
        data_dbs2.append(data_dbs2_load[0])
        data_dbs3.append(data_dbs3_load[0])
        data_dbs4.append(data_dbs4_load[0])
        data_dbs5.append(data_dbs5_load[0])
        data_dbs6.append(data_dbs6_load[0])

    # concatenate data
    data_dbs1 = np.array(data_dbs1).T
    data_dbs2 = np.array(data_dbs2).T
    data_dbs3 = np.array(data_dbs3).T
    data_dbs4 = np.array(data_dbs4).T
    data_dbs5 = np.array(data_dbs5).T
    data_dbs6 = np.array(data_dbs6).T

    data_dbs = [data_dbs1, data_dbs2, data_dbs3, data_dbs4, data_dbs5, data_dbs6]

    ######################## Dataframe #########################

    dbs_states = [
        "dbs-off",
        "suppression",
        "efferent",
        "afferent",
        "passing-fibres",
        "dbs-comb",
    ]
    populations = [
        "IT",
        "StrD1",
        "StrD2",
        "STN",
        "GPi",
        "GPe",
        "Thal",
        "PFC",
        "StrThal",
    ]

    subject_id = []
    population_id = []
    dbs_id = []
    dbs_state = []
    population = []
    rate = []

    # lists for dataframe
    for i, state in enumerate(dbs_states):
        for j, pop in enumerate(populations):
            for k in range(len(data_dbs[0][0])):
                subject_id.append(k)
                population_id.append(j)
                dbs_id.append(i)
                dbs_state.append(state)
                population.append(pop)
                rate.append(data_dbs[i][j][k])

    # dataframe
    data_df = pd.DataFrame(
        {
            "subject_id": subject_id,
            "population_id": population_id,
            "dbs_id": dbs_id,
            "dbs_state": dbs_state,
            "population": population,
            "rate": rate,
        }
    )

    # save dataframe
    filename = "statistic/regression_activity_change"
    save_table(data_df, filename)

    ##################### statistic mixed-effects model ################
    for population_id, population in enumerate(data_df["population"].unique()):
        data_df_pop = data_df[data_df["population"] == population]
        # fit a mixed-effects model
        model = smf.mixedlm(
            "rate ~ C(dbs_state, Treatment('dbs-off'))",
            data_df_pop,
            groups=data_df_pop["subject_id"],
        )

        result = model.fit()

        # get the p-values for the coefficients and correct them for multiple comparisons
        p_values = result.pvalues
        # exlude Group Var and Intercept
        p_values = p_values.drop("Group Var")
        p_values = p_values.drop("Intercept")
        p_values_corrected = multipletests(p_values, method="bonferroni")[:2]
        p_values_corrected_df = pd.DataFrame(
            {
                "p-values uncor": p_values,
                "p-values cor": p_values_corrected[1],
                "reject": p_values_corrected[0],
            }
        )
        # also add the corresponding  coefficients, Std.Err., z values, and confidence intervalls to the dataframe
        further_columns = {
            "coefficients": result.params,
            "Std.Err.": result.bse,
            "z values": result.tvalues,
            "[0.025": result.conf_int()[0],
            "0.975]": result.conf_int()[1],
        }
        for column_name, column_data in further_columns.items():
            further_columns[column_name] = column_data.drop("Group Var")
            further_columns[column_name] = further_columns[column_name].drop(
                "Intercept"
            )
        p_values_corrected_df = pd.concat(
            [p_values_corrected_df, pd.DataFrame(further_columns)], axis=1
        )

        # save the p_values_corrected_df as csv
        p_values_corrected_df.round(3).to_csv(
            f"statistic/regression_activity_change_{population}.csv",
            decimal=",",
        )

        # print summary
        with open(
            "statistic/regression_activity_change.txt",
            ["a", "w"][int(population_id == 0)],
        ) as f:
            f.write(f"\n\n{population}\n")
            f.write(result.summary().as_text())


#####################################################################################################
######################################## call test function #########################################
#####################################################################################################

# if __name__ == "__main__":
# Funktion aufrufen
# manova_change_activity()
# anova_change_activity()
# anova_load_simulation(100)
# linear_regression()
# run_statistic(False, True, False, 100)
