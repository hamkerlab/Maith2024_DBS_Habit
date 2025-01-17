import matplotlib.pyplot as plt
import numpy as np
import statistic as stat
from CompNeuroPy import load_variables
import pandas as pd
import seaborn as sns
import pingouin as pg
from statistic import load_data_previously_selected
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

#################################################################################################################
########################################### plot figures ########################################################
#################################################################################################################

__fig_shortcut_on_off_line__ = False
__fig_shortcut_on_off__ = False
__fig_dbs_on_off_14_and_100__ = False
__fig_activity_changes_dbs_on__ = False
__fig_activity_changes_dbs_off__ = False
__fig_gpi_scatter__ = False
__fig_load_simulate__ = False
__fig_load_simulate_dbscomb__ = False
__fig_dbs_parameter__ = False
__fig_parameter_gpi_inhib__ = False
__fig_weights_over_time__ = False
__fig_support_over_time__ = False


##############################################################################
############################# scale y-axis ###################################
##############################################################################

min_y_reward = 0
max_y_reward = 30

min_y_habit = 0
max_y_habit = 30

label_size = 9

lighter_darkblue = (0, 0, 0.65)

#################################################################################################################
##################################### __fig_shortcut_on_off__ ###################################################
#################################################################################################################


def shortcut_on_off(switch, number_of_simulations):

    ################################################# load data #################################################
    # load simulation data
    filepath1 = "data/simulation_data/Results_Shortcut0_DBS_State0.json"
    filepath2 = "data/simulation_data/Results_Shortcut1_DBS_State0.json"

    result1 = stat.read_json_data(filepath1)
    result2 = stat.read_json_data(filepath2)
    if switch:
        result1 = stat.processing_habit_data(result1, number_of_simulations)
        result2 = stat.processing_habit_data(result2, number_of_simulations)
    else:
        result1 = stat.processing_data(result1, number_of_simulations)
        result2 = stat.processing_data(result2, number_of_simulations)

    # load patient data
    filepath1 = "data/patient_data/RewardsPerSession_ON.json"
    filepath2 = "data/patient_data/RewardsPerSession_OFF.json"

    # delete nan rows and switch rewarded to unrewarded decision
    result3 = stat.read_json_data(filepath2)
    result3 = result3[~np.isnan(result3).any(axis=1)]
    result3 = 40 - result3

    ################################################## Data Point Cloud ##########################################

    data = [
        result3.T,
        result2.T,
        result1.T,
    ]

    data = np.array(data)

    """
    ############################################## means ###############################################
    # mean simulation data
    mean1 = stat.mean_data(result1)
    mean2 = stat.mean_data(result2)

    # mean patient data
    mean_sessions = stat.mean_dbs_session(filepath1, filepath2, switch)
    meanOFF = mean_sessions[1]

    # mean bars
    means = [meanOFF, mean2, mean1]

    ######################################### standard errors ##########################################
    # standard error simualtion data
    standarderror1 = stat.standarderror_data(result1)
    standarderror2 = stat.standarderror_data(result2)

    # standard error patient data
    standarderror = stat.session_standard_error(filepath1, filepath2, switch)
    standarderrorOFF = standarderror[1]

    # standarderrors bars
    standarderrors = [standarderrorOFF, standarderror2, standarderror1]
    """

    ############################################### histo settings ##############################################

    # sessions
    sessions = ["1", "2", "3"]
    x = np.arange(len(sessions)) * 1.5

    # bar width
    width = 0.3

    # bar colors
    colors = [lighter_darkblue, "steelblue", "lightblue"]

    # bar positions
    positions = [x - 1.3 * width, x, x + 1.3 * width]

    labels = ["DBS OFF Patients", "Model Plastic Shortcut", "Model Fixed Shortcut"]

    fig, ax = plt.subplots(figsize=(3.4, 3.4))

    ################################################## boxplots data #################################################

    for i in range(len(data)):
        # boxplot for each category and session
        bp = ax.boxplot(
            data[i].T,
            positions=positions[i],
            widths=0.3,
            patch_artist=True,
            showmeans=False,
            meanline=False,
            meanprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "black",
                "markersize": 3,
                "linestyle": "--",
                "linewidth": 1,
            },
            boxprops={"color": "black"},
            medianprops={"color": "black"},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            flierprops={
                "marker": "o",
                "color": "black",
                "markersize": 3,
                "markeredgecolor": "black",
            },
        )

        bp["boxes"][0].set_label(labels[i])

        # boxplot color
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_edgecolor("black")

    # legend
    ax.legend(fontsize=label_size)  # small

    ################################################### significance * #############################################

    # function for significance
    def add_star(ax, x1, x2, y):
        """Add asterisks for significant differences."""
        y_offset = 0.5
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + y_offset, y + y_offset, y],
            color="black",
            linewidth=1,
        )
        ax.text((x1 + x2) / 2, y + y_offset, "*", fontsize=12, ha="center")

    # star position
    add_star(ax, 3, 3.4, 22)

    ################################################### axis settings ###############################################

    ax.tick_params(axis="both", labelsize=label_size)

    # x-axis
    plt.xticks(x, sessions)

    # y-axis
    plt.xlabel("Session", fontweight="bold", fontsize=label_size)
    if switch:
        plt.ylabel("unrewarded decisions", fontweight="bold", fontsize=label_size)
    else:
        plt.ylabel("rewards", fontweight="bold", fontsize=label_size)

    # y-axis min/max
    if switch:
        plt.ylim(min_y_habit, max_y_habit)
    else:
        plt.ylim(min_y_reward, max_y_reward)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[0, -0.009, 1, 1],
    )

    # save fig
    plt.savefig("fig/__fig_shortcut_on_off__.png", dpi=300)
    plt.savefig("fig/__fig_shortcut_on_off__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
############################### __appendix__fig_shortcut_on_off_line__ ##########################################
#################################################################################################################


def shortcut_on_off_line(number_of_simulations):

    ################################################# load data #################################################
    # simulation data
    filepath1 = "data/simulation_data/Results_Shortcut0_DBS_State0.json"
    filepath2 = "data/simulation_data/Results_Shortcut1_DBS_State0.json"

    # patient data
    filepath3 = "data/patient_data/RewardsPerSession_OFF_line.json"

    result1 = stat.read_json_data(filepath1)
    result2 = stat.read_json_data(filepath2)
    result3 = stat.read_json_data(filepath3)

    # processing data
    result1 = stat.processing_line(result1, number_of_simulations)
    result2 = stat.processing_line(result2, number_of_simulations)
    result3 = stat.processing_line(result3, number_of_simulations)

    ############################################## means ###############################################
    # mean simulation data
    mean1 = stat.mean_data_line(result1)
    mean2 = stat.mean_data_line(result2)
    mean3 = stat.mean_data_line(result3)

    # mean bars
    means = [mean3, mean2, mean1]

    ############################################### line plot settings ##############################################

    # x-Achse in 5er Schritten von 0 bis 120
    x_values = np.arange(0, 120, 5)

    # bar colors
    colors = [lighter_darkblue, "steelblue", "lightblue"]

    labels = ["DBS OFF Patients", "Model Plastic Shortcut", "Model Fixed Shortcut"]

    linestyle = ["-", ":", "--"]

    fig, ax = plt.subplots(figsize=(4, 3.4))

    ################################################## plot lines #################################################

    for i in range(len(means)):
        plt.plot(
            x_values + 2.5,
            means[i],
            color=colors[i],
            label=labels[i],
            linestyle=linestyle[i],
        )

    ################################################### axis settings ###############################################

    # means session3 patient data
    plt.axvline(x=40, color="black", linestyle="-", linewidth=1.0)
    plt.axvline(x=80, color="black", linestyle="-", linewidth=1.0)
    plt.axvline(
        x=60, color="red", linestyle="--", linewidth=1.0, label="Reward Reversal"
    )

    # plot legend
    plt.legend(fontsize=label_size)  # small

    # x-axis
    plt.xticks(range(0, 121, 20))
    ax.set_xlim(left=0, right=120)

    # intervall-labels
    plt.text(
        20,
        -0.8,
        "Session1",
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
        fontsize=label_size,
    )
    plt.text(
        60,
        -0.8,
        "Session2",
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
        fontsize=label_size,
    )
    plt.text(
        100,
        -0.8,
        "Session3",
        ha="center",
        va="center",
        fontweight="bold",
        color="black",
        fontsize=label_size,
    )

    # y-axis
    plt.ylim(0, 7)
    plt.ylabel("unrewarded decisions", fontweight="bold", fontsize=label_size)

    plt.tick_params(axis="both", labelsize=label_size)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[0, 0.007, 1, 1],
    )

    # save fig
    plt.savefig("fig/__fig_shortcut_on_off_line__.png", dpi=300)
    plt.savefig("fig/__fig_shortcut_on_off_line__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
################################## __fig_dbs_on_off_14_and_100__ ################################################
#################################################################################################################


def dbs_on_off_14_and_100(switch=True, shortcut=True):
    """
    Args:
    switch: bool
        if True use data from unrewarded decisions, else not sure maybe rewarded decisions
    shortcut: bool
        if True use data from plastic shortcut, else fixed shortcut
    """

    ################################################# load data #################################################
    # load simulation data
    filepath1 = f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State1.json"
    filepath2 = f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State2.json"
    # filepath3 = "data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State3.json"
    filepath4 = f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State4.json"
    filepath5 = f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State5.json"
    filepath6 = f"data/simulation_data/Results_Shortcut{int(shortcut)}_DBS_State0.json"

    result1 = stat.read_json_data(filepath1)
    result2 = stat.read_json_data(filepath2)
    # result3 = stat.read_json_data(filepath3)
    result4 = stat.read_json_data(filepath4)
    result5 = stat.read_json_data(filepath5)
    result6 = stat.read_json_data(filepath6)

    if switch == False:
        result_14_1 = stat.processing_data(result1, 14)
        result_14_2 = stat.processing_data(result2, 14)
        # result_14_3 = stat.processing_data(result3, 14)
        result_14_4 = stat.processing_data(result4, 14)
        result_14_5 = stat.processing_data(result5, 14)
        result_14_6 = stat.processing_data(result6, 14)

        result_100_1 = stat.processing_data(result1, 100)
        result_100_2 = stat.processing_data(result2, 100)
        # result_100_3 = stat.processing_data(result3, 100)
        result_100_4 = stat.processing_data(result4, 100)
        result_100_5 = stat.processing_data(result5, 100)
        result_100_6 = stat.processing_data(result6, 100)
    else:
        result_14_1 = stat.processing_habit_data(result1, 14)
        result_14_2 = stat.processing_habit_data(result2, 14)
        # result_14_3 = stat.processing_habit_data(result3, 14)
        result_14_4 = stat.processing_habit_data(result4, 14)
        result_14_5 = stat.processing_habit_data(result5, 14)
        result_14_6 = stat.processing_habit_data(result6, 14)

        result_100_1 = stat.processing_habit_data(result1, 100)
        result_100_2 = stat.processing_habit_data(result2, 100)
        # result_100_3 = stat.processing_habit_data(result3, 100)
        result_100_4 = stat.processing_habit_data(result4, 100)
        result_100_5 = stat.processing_habit_data(result5, 100)
        result_100_6 = stat.processing_habit_data(result6, 100)

    # load patient data
    filepath1 = "data/patient_data/RewardsPerSession_ON.json"
    filepath2 = "data/patient_data/RewardsPerSession_OFF.json"

    # delete nan rows and switch rewarded to unrewarded decision
    resultON = stat.read_json_data(filepath1)
    resultOFF = stat.read_json_data(filepath2)
    resultON = resultON[~np.isnan(resultOFF).any(axis=1)]
    resultOFF = resultOFF[~np.isnan(resultOFF).any(axis=1)]
    resultON = 40 - resultON
    resultOFF = 40 - resultOFF

    ################################################## Data Point Cloud ##########################################

    data1 = [
        resultON.T,
        resultOFF.T,
    ]

    data2 = [
        result_14_1.T,
        result_14_2.T,
        result_14_4.T,
        result_14_5.T,
        result_14_6.T,
    ]

    data3 = [
        result_100_1.T,
        result_100_2.T,
        result_100_4.T,
        result_100_5.T,
        result_100_6.T,
    ]

    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    """
    ############################################## means ###############################################
    # mean simulation data
    mean_14_1 = stat.mean_data(result_14_1)
    mean_14_2 = stat.mean_data(result_14_2)
    # mean_14_3 = stat.mean_data(result_14_3)
    mean_14_4 = stat.mean_data(result_14_4)
    mean_14_5 = stat.mean_data(result_14_5)
    mean_14_6 = stat.mean_data(result_14_6)

    mean_100_1 = stat.mean_data(result_100_1)
    mean_100_2 = stat.mean_data(result_100_2)
    # mean_100_3 = stat.mean_data(result_100_3)
    mean_100_4 = stat.mean_data(result_100_4)
    mean_100_5 = stat.mean_data(result_100_5)
    mean_100_6 = stat.mean_data(result_100_6)

    # mean patient data
    mean_sessions = stat.mean_dbs_session(filepath1, filepath2, switch)
    meanON = mean_sessions[0]
    meanOFF = mean_sessions[1]

    # mean bars
    means_14 = [mean_14_1, mean_14_2, mean_14_4, mean_14_5, mean_14_6]
    means_100 = [mean_100_1, mean_100_2, mean_100_4, mean_100_5, mean_100_6]
    means = [meanON, meanOFF]

    ######################################### standard errors ##########################################
    # standard error simualtion data
    standarderror_14_1 = stat.standarderror_data(result_14_1)
    standarderror_14_2 = stat.standarderror_data(result_14_2)
    # standarderror_14_3 = stat.standarderror_data(result_14_3)
    standarderror_14_4 = stat.standarderror_data(result_14_4)
    standarderror_14_5 = stat.standarderror_data(result_14_5)
    standarderror_14_6 = stat.standarderror_data(result_14_6)

    standarderror_100_1 = stat.standarderror_data(result_100_1)
    standarderror_100_2 = stat.standarderror_data(result_100_2)
    # standarderror_100_3 = stat.standarderror_data(result_100_3)
    standarderror_100_4 = stat.standarderror_data(result_100_4)
    standarderror_100_5 = stat.standarderror_data(result_100_5)
    standarderror_100_6 = stat.standarderror_data(result_100_6)

    # standard error patient data
    standarderror = stat.session_standard_error(filepath1, filepath2, switch)
    standarderrorON = standarderror[0]
    standarderrorOFF = standarderror[1]

    # standarderrors bars
    standarderror_14 = [
        standarderror_14_1,
        standarderror_14_2,
        standarderror_14_4,
        standarderror_14_5,
        standarderror_14_6,
    ]
    standarderror_100 = [
        standarderror_100_1,
        standarderror_100_2,
        standarderror_100_4,
        standarderror_100_5,
        standarderror_100_6,
    ]
    standarderror = [standarderrorON, standarderrorOFF]
    """

    ############################################### histo settings ##############################################

    # sessions
    session = ["1", "2", "3"]
    x = np.arange(len(session)) * 2

    # bar width
    width = 0.2

    # bar colors
    patient_colors = [(0.8, 0, 0, 0.7), lighter_darkblue]
    simulation_colors = [
        (1, 0.7, 0.7, 0.8),  # very bright red
        (1, 0.5, 0.5, 0.8),  # light red
        (1, 0.3, 0.3, 0.8),  # red
        (0.8, 0, 0, 0.8),  # dark red
        lighter_darkblue,
    ]

    # bar positions
    patient_positions = [x - 0.75 * width, x + 0.75 * width]

    simulation_positions = [
        x - 3 * width,
        x - 1.5 * width,
        x,
        x + 1.5 * width,
        x + 3 * width,
    ]

    patient_labels = ["dbs-on", "dbs-off"]
    simulation_labels = [
        "suppression",
        "efferent",
        "passing-fibres",
        "dbs-comb",
        # "dbs-off",
    ]

    # plot size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7.5))

    ################################################## plot boxplots patient data #################################################

    for i in range(len(patient_positions)):  # dbs-states
        for session_idx in range(len(session)):  # sessions

            values = data1[i, session_idx]
            values = values[~np.isnan(values)]

            bp = ax1.boxplot(
                values,
                positions=[patient_positions[i][session_idx]],
                widths=width,
                patch_artist=True,
                showmeans=False,
                meanline=False,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                    "markersize": 3,
                    "linestyle": "--",
                    "linewidth": 1,
                },
                boxprops={"color": "black"},
                medianprops={"color": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                flierprops={
                    "marker": "o",
                    "color": "black",
                    "markersize": 3,
                    "markeredgecolor": "black",
                },
            )

            # boxplot color
            for patch in bp["boxes"]:
                patch.set_facecolor(patient_colors[i])
                patch.set_edgecolor("black")

        bp["boxes"][0].set_label(patient_labels[i])

    # plot legend
    ax1.legend(fontsize=label_size, loc="upper left")  # x-small

    # ax1.set_xlabel("session")
    ax1.set_ylim(0, 41)
    ax1.tick_params(axis="both", which="major", labelsize=label_size)
    if switch:
        ax1.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax1.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ########################################## plot boxplots simulation data N=14 #################################################

    for i in range(len(simulation_positions)):  # dbs_states
        for session_idx in range(len(session)):  # sessions

            values = data2[i, session_idx]
            values = values[~np.isnan(values)]

            bp = ax2.boxplot(
                values,
                positions=[simulation_positions[i][session_idx]],
                widths=width,
                patch_artist=True,
                showmeans=False,
                meanline=False,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                    "markersize": 3,
                    "linestyle": "--",
                    "linewidth": 1,
                },
                boxprops={"color": "black"},
                medianprops={"color": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                flierprops={
                    "marker": "o",
                    "color": "black",
                    "markersize": 3,
                    "markeredgecolor": "black",
                },
            )

            # boxplot color
            for patch in bp["boxes"]:
                patch.set_facecolor(simulation_colors[i])
                patch.set_edgecolor("black")

        if i < 4:
            bp["boxes"][0].set_label(simulation_labels[i])

    # plot legend
    ax2.legend(fontsize=label_size, loc="upper left")  # x-small
    ax2.set_ylim(0, 41)
    ax2.tick_params(axis="both", which="major", labelsize=label_size)
    if switch:
        ax2.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax2.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ############################################ plot boxplots simulation data N=100 #################################################

    for i in range(len(simulation_positions)):  # dbs-states
        for session_idx in range(len(session)):  # sessions

            values = data3[i, session_idx]
            values = values[~np.isnan(values)]

            bp = ax3.boxplot(
                values,
                positions=[simulation_positions[i][session_idx]],
                widths=width,
                patch_artist=True,
                showmeans=False,
                meanline=False,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "black",
                    "markersize": 3,
                    "linestyle": "--",
                    "linewidth": 1,
                },
                boxprops={"color": "black"},
                medianprops={"color": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                flierprops={
                    "marker": "o",
                    "color": "black",
                    "markersize": 3,
                    "markeredgecolor": "black",
                },
            )

            # boxplot color
            for patch in bp["boxes"]:
                patch.set_facecolor(simulation_colors[i])
                patch.set_edgecolor("black")

    # settings axis
    ax3.set_xlabel("Session", fontweight="bold", fontsize=label_size)
    ax3.set_ylim(0, 41)
    ax3.tick_params(axis="both", which="major", labelsize=label_size)
    if switch:
        ax3.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax3.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ################################################### significance * #############################################

    # function for significance
    def add_star(ax, x1, x2, y):
        """Add asterisks for significant differences"""
        y_offset = 0.5
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + y_offset, y + y_offset, y],
            color="black",
            linewidth=1,
        )
        ax.text((x1 + x2) / 2, y - 0.2, "*", fontsize=10, ha="center")

    if shortcut == 1:
        add_star(ax3, 4.3, 4.6, 34)
        add_star(ax3, 3.7, 4.6, 36)
        add_star(ax3, 3.4, 4.6, 38)

    ################################################### axis settings ###############################################

    # same x-axis ticks for all subplots
    plt.setp([ax1, ax2, ax3], xticks=x, xticklabels=session)

    # same x-axis in all plots
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.set_xticks(x)
    ax3.set_xticklabels(session)

    ax1.set_xlim(ax3.get_xlim())
    ax2.set_xlim(ax3.get_xlim())
    ax3.set_xlim(ax1.get_xlim())

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.004, -0.004, 1, 1],
    )

    # plot labels
    plt.text(0.01, 0.98, "A", transform=plt.gcf().transFigure, fontsize=11)
    plt.text(0.01, 0.655, "B", transform=plt.gcf().transFigure, fontsize=11)
    plt.text(0.01, 0.33, "C", transform=plt.gcf().transFigure, fontsize=11)

    # save fig
    plt.savefig(
        f"fig/__fig_dbs_on_off_14_and_100_Shortcut{int(shortcut)}__.png", dpi=300
    )
    plt.savefig(
        f"fig/__fig_dbs_on_off_14_and_100_Shortcut{int(shortcut)}__.pdf",
        format="pdf",
        dpi=300,
    )
    plt.close("all")


#################################################################################################################
#################################### __fig_activity_changes_dbs_on__ ############################################
#################################################################################################################


def activity_changes_dbs_on():

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
            data_dbs1_load = stat.read_json_data(filepath1 + f"_id{i}.json")
            data_dbs2_load = stat.read_json_data(filepath2 + f"_id{i}.json")
            data_dbs3_load = stat.read_json_data(filepath3 + f"_id{i}.json")
            data_dbs4_load = stat.read_json_data(filepath4 + f"_id{i}.json")
            data_dbs5_load = stat.read_json_data(filepath5 + f"_id{i}.json")
            data_dbs6_load = stat.read_json_data(filepath6 + f"_id{i}.json")

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

        # standard error
        table_error_dbs1 = []
        table_error_dbs2 = []
        table_error_dbs3 = []
        table_error_dbs4 = []
        table_error_dbs5 = []
        table_error_dbs6 = []

        for i in range(len(data_dbs1)):
            table_error_dbs1.append(np.std(data_dbs1[i]))
            table_error_dbs2.append(np.std(data_dbs2[i]))
            table_error_dbs3.append(np.std(data_dbs3[i]))
            table_error_dbs4.append(np.std(data_dbs4[i]))
            table_error_dbs5.append(np.std(data_dbs5[i]))
            table_error_dbs6.append(np.std(data_dbs6[i]))

        table_legend = [
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

        table = {
            " ": table_legend,
            "mean_dbs-off": table_mean_dbs1,
            "error_dbs-off": table_error_dbs1,
            "mean_supression": table_mean_dbs2,
            "error_supression": table_error_dbs2,
            "mean_efferent": table_mean_dbs3,
            "error_efferent": table_error_dbs3,
            "mean_afferent": table_mean_dbs4,
            "error_afferent": table_error_dbs4,
            "mean_passing-fibres": table_mean_dbs5,
            "error_passing-fibres": table_error_dbs5,
            "mean_dbs-comb": table_mean_dbs6,
            "error_dbs-comb": table_error_dbs6,
        }

        filename = "statistic/mean_error_table_fig4"
        stat.save_table(table, filename)

        ########################################### difference to dbs-off ###############################################

        for i in range(len(data_dbs1)):
            for j in range(len(data_dbs1[0][0])):
                if data_dbs1[i][0][j] != 0:
                    data_dbs2[i][0][j] = data_dbs2[i][0][j] - data_dbs1[i][0][j]
                    data_dbs3[i][0][j] = data_dbs3[i][0][j] - data_dbs1[i][0][j]
                    data_dbs4[i][0][j] = data_dbs4[i][0][j] - data_dbs1[i][0][j]
                    data_dbs5[i][0][j] = data_dbs5[i][0][j] - data_dbs1[i][0][j]
                    data_dbs6[i][0][j] = data_dbs6[i][0][j] - data_dbs1[i][0][j]

        ####################################### mean / standarderror N=100 #############################################

        # means
        mean_dbs1 = []
        mean_dbs2 = []
        mean_dbs3 = []
        mean_dbs4 = []
        mean_dbs5 = []
        mean_dbs6 = []

        for i in range(len(data_dbs1)):
            mean_dbs1.append(np.mean(data_dbs1[i]))
            mean_dbs2.append(np.mean(data_dbs2[i]))
            mean_dbs3.append(np.mean(data_dbs3[i]))
            mean_dbs4.append(np.mean(data_dbs4[i]))
            mean_dbs5.append(np.mean(data_dbs5[i]))
            mean_dbs6.append(np.mean(data_dbs6[i]))

        # standard error
        error_dbs1 = []
        error_dbs2 = []
        error_dbs3 = []
        error_dbs4 = []
        error_dbs5 = []
        error_dbs6 = []

        for i in range(len(data_dbs1)):
            error_dbs1.append(
                # np.std(data_dbs1[i], ddof=1) / np.sqrt(np.size(data_dbs1[i]))
                np.std(data_dbs1[i])
            )
            error_dbs2.append(np.std(data_dbs2[i]))
            error_dbs3.append(np.std(data_dbs3[i]))
            error_dbs4.append(np.std(data_dbs4[i]))
            error_dbs5.append(np.std(data_dbs5[i]))
            error_dbs6.append(np.std(data_dbs6[i]))

        ############################################ histo settings #####################################################

        # sessions
        x = 1

        # bar width
        width = 0.5

        colors = [(1.0, 0.2, 0.2, 0.7)]

        # bar positions
        positions = [
            x + 6 * width,
            x + 4.5 * width,
            x + 3 * width,
            x + 1.5 * width,
            x,
            x - 1.5 * width,
            x - 3 * width,
            x - 4.5 * width,
            x - 6 * width,
        ]

        labels = [
            "suppression",
            "efferent",
            "passing-fibres",
            "dbs-comb",
            # "dbs-off",
        ]

        label_y = [
            "$\\mathbf{Cor_{in}}$",
            "$\\mathbf{StrD1}$",
            "$\\mathbf{StrD2}$",
            "$\\mathbf{STN}$",
            "$\\mathbf{GPi}$",
            "$\\mathbf{GPe}$",
            "$\\mathbf{Thalamus}$",
            "$\\mathbf{Cor_{dec}}$",
            "$\\mathbf{StrThal}$",
        ]

        # plot size
        fig, axs = plt.subplots(1, 5, figsize=(6.4, 2.5))

        xlim_neg = -0.1
        xlim_pos = 0.1

        ################################################## plot bars #################################################

        ####################### suppression data ################################
        for i in range(len(mean_dbs2)):
            # plot means
            axs[0].barh(
                positions[i],
                mean_dbs2[i],
                height=width,
                color=colors,
            )

            for i in range(len(error_dbs2)):
                # plot error bars
                axs[0].errorbar(
                    x=mean_dbs2[i],
                    y=positions[i],
                    xerr=error_dbs2[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        axs[0].axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs[0].set_yticks(positions)
        axs[0].set_yticklabels(label_y, fontsize=label_size)

        # x-axis
        axs[0].set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
        axs[0].set_xticklabels(["-0.1", " ", "0", " ", "0.1"])
        axs[0].set_xlabel(
            "difference to\ndbs-off", fontweight="bold", fontsize=label_size
        )
        axs[0].set_xlim(xlim_neg, xlim_pos)
        # axs[0].set_xticks([-0.1, -0.05, 0, 0.05, 0.1])

        # title
        axs[0].set_title("suppression", fontweight="bold", fontsize=label_size)

        ############# significance* ####################
        for i in range(len(mean_dbs2)):
            if i < 4:
                continue

            if mean_dbs2[i] >= 0:
                offset = +0.008
                rot = -90
            else:
                if i == 8:
                    offset = -0.01
                else:
                    offset = -0.008
                rot = 90

            axs[0].text(
                x=mean_dbs2[i] + offset,  # leicht rechts des Balken-Endes
                y=positions[i],  # exakt auf der y-Position des Balkens
                s="*",  # Stern als Text
                fontsize=10,  # Stern-Größe
                color="black",  # Stern-Farbe
                ha="center",  # Zentrierung horizontal
                va="center",  # Zentrierung vertikal
                rotation=rot,
            )

        ####################### efferent data ################################
        for i in range(len(mean_dbs3)):
            # plot means
            axs[1].barh(
                positions[i],
                mean_dbs3[i],
                height=width,
                color=colors,
            )

            # plot legend
            # axs[0].legend(labels, fontsize="large", loc="upper left")

            for i in range(len(error_dbs3)):
                # plot error bars
                axs[1].errorbar(
                    x=mean_dbs3[i],
                    y=positions[i],
                    xerr=error_dbs3[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        axs[1].axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs[1].set_yticks(positions)
        axs[1].set_yticklabels([])

        # x-axis
        axs[1].set_xlim(xlim_neg, xlim_pos)
        axs[1].set_xticks(np.linspace(xlim_neg, xlim_pos, 5))
        axs[1].set_xticklabels([""] * 5)  # Entferne die Labels

        axs[1].tick_params(axis="both", labelsize=label_size)

        # title
        axs[1].set_title("efferent", fontweight="bold", fontsize=label_size)

        ############# significance* ####################
        for i in range(len(mean_dbs3)):
            if i < 4:
                continue

            if mean_dbs3[i] >= 0:
                offset = +0.008
                rot = -90
            else:
                offset = -0.008
                rot = 90

            axs[1].text(
                x=mean_dbs3[i] + offset,  # leicht rechts des Balken-Endes
                y=positions[i],  # exakt auf der y-Position des Balkens
                s="*",  # Stern als Text
                fontsize=10,  # Stern-Größe
                color="black",  # Stern-Farbe
                ha="center",  # Zentrierung horizontal
                va="center",  # Zentrierung vertikal
                rotation=rot,
            )

        ####################### afferent data ################################
        for i in range(len(mean_dbs4)):
            # plot means
            axs[2].barh(
                positions[i],
                mean_dbs4[i],
                height=width,
                color=colors,
            )

            # plot legend
            # axs[0].legend(labels, fontsize="large", loc="upper left")

            for i in range(len(error_dbs4)):
                # plot error bars
                axs[2].errorbar(
                    x=mean_dbs4[i],
                    y=positions[i],
                    xerr=error_dbs4[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        axs[2].axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs[2].set_yticks(positions)
        axs[2].set_yticklabels([])

        # x-axis
        axs[2].set_xlim(xlim_neg, xlim_pos)
        axs[2].set_xticks(np.linspace(xlim_neg, xlim_pos, 5))
        axs[2].set_xticklabels([""] * 5)  # Entferne die Labels

        axs[2].tick_params(axis="both", labelsize=label_size)

        # title
        axs[2].set_title("afferent", fontweight="bold", fontsize=label_size)

        ############# significance* ####################
        for i in range(len(mean_dbs4)):
            if i < 4:
                continue

            if mean_dbs4[i] >= 0:
                if i == 8:
                    offset = +0.013
                else:
                    offset = +0.008

                rot = -90
            else:
                if i == 4:
                    offset = -0.005
                else:
                    offset = -0.008

                rot = 90

            axs[2].text(
                x=mean_dbs4[i] + offset,  # leicht rechts des Balken-Endes
                y=positions[i],  # exakt auf der y-Position des Balkens
                s="*",  # Stern als Text
                fontsize=10,  # Stern-Größe
                color="black",  # Stern-Farbe
                ha="center",  # Zentrierung horizontal
                va="center",  # Zentrierung vertikal
                rotation=rot,
            )

        ####################### passing fibres data ################################
        for i in range(len(mean_dbs5)):
            # plot means
            axs[3].barh(
                positions[i],
                mean_dbs5[i],
                height=width,
                color=colors,
            )

            for i in range(len(error_dbs5)):
                # plot error bars
                axs[3].errorbar(
                    x=mean_dbs5[i],
                    y=positions[i],
                    xerr=error_dbs5[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        axs[3].axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs[3].set_yticks(positions)
        axs[3].set_yticklabels([])

        # x-axis
        axs[3].set_xlim(xlim_neg, xlim_pos)
        axs[3].set_xticks(np.linspace(xlim_neg, xlim_pos, 5))
        axs[3].set_xticklabels([""] * 5)  # Entferne die Labels

        axs[3].tick_params(axis="both", labelsize=label_size)

        # title
        axs[3].set_title("passing-fibres", fontweight="bold", fontsize=label_size)

        ############# significance* ####################
        for i in range(len(mean_dbs5)):
            if i < 3 or i > 7:
                continue

            if mean_dbs5[i] >= 0:
                offset = +0.008
                rot = -90
            else:
                offset = -0.008
                rot = 90

            axs[3].text(
                x=mean_dbs5[i] + offset,  # leicht rechts des Balken-Endes
                y=positions[i],  # exakt auf der y-Position des Balkens
                s="*",  # Stern als Text
                fontsize=10,  # Stern-Größe
                color="black",  # Stern-Farbe
                ha="center",  # Zentrierung horizontal
                va="center",  # Zentrierung vertikal
                rotation=rot,
            )

        ####################### dbs-comb data ################################
        for i in range(len(mean_dbs6)):
            # plot means
            axs[4].barh(
                positions[i],
                mean_dbs6[i],
                height=width,
                color=colors,
            )

            for i in range(len(error_dbs6)):
                # plot error bars
                axs[4].errorbar(
                    x=mean_dbs6[i],
                    y=positions[i],
                    xerr=error_dbs6[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        axs[4].axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs[4].set_yticks(positions)
        axs[4].set_yticklabels([])

        # x-axis
        axs[4].set_xlim(xlim_neg, xlim_pos)
        axs[4].set_xticks(np.linspace(xlim_neg, xlim_pos, 5))
        axs[4].set_xticklabels([""] * 5)  # Entferne die Labels

        axs[4].tick_params(axis="both", labelsize=label_size)

        # title
        axs[4].set_title("dbs-comb", fontweight="bold", fontsize=label_size)

        ############# significance* ####################
        for i in range(len(mean_dbs6)):
            if i < 3 or i == 5:
                continue

            if mean_dbs6[i] >= 0:
                offset = +0.008
                rot = -90
            else:
                if i == 4:
                    offset = -0.003
                elif i == 8:
                    offset = -0.010
                else:
                    offset = -0.008

                rot = 90

            axs[4].text(
                x=mean_dbs6[i] + offset,  # leicht rechts des Balken-Endes
                y=positions[i],  # exakt auf der y-Position des Balkens
                s="*",  # Stern als Text
                fontsize=10,  # Stern-Größe
                color="black",  # Stern-Farbe
                ha="center",  # Zentrierung horizontal
                va="center",  # Zentrierung vertikal
                rotation=rot,
            )

        # Adjust layout to minimize margins
        # plt.subplots_adjust(left=0.07, right=0.96, top=0.9, bottom=0.15, wspace=0.2)

        # Adjust layout
        plt.tight_layout(
            pad=0,
            h_pad=1.08,
            w_pad=1.08,
            rect=[-0.002, -0.02, 1, 1],
        )

        # save fig
        if session == 0:
            plt.savefig("fig/__fig_activity_change_dbs_on_init__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_on_init__.pdf", format="pdf", dpi=300
            )
        else:
            plt.savefig("fig/__fig_activity_change_dbs_on_learn__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_on_learn__.pdf", format="pdf", dpi=300
            )

        plt.close("all")


#################################################################################################################
#################################### __fig_activity_changes_dbs_off__ ############################################
#################################################################################################################


def activity_changes_dbs_off():

    for session in range(1, 2):
        # initial
        number_of_simulations = 100
        data_dbs1 = []

        ################################################# load data #################################################
        filepath1 = f"data/activity_change/activity_change_dbs_state0_session{session}"

        for i in range(number_of_simulations):
            data_dbs1_load = stat.read_json_data(filepath1 + f"_id{i}.json")

            # append loaded data to list
            data_dbs1.append(data_dbs1_load)

        # concatenate data
        data_dbs1 = np.array(data_dbs1).T

        # print("\n", data_dbs1, "\n")

        ####################################### mean / standarderror N=100 #############################################

        # means
        mean_dbs1 = []

        for i in range(len(data_dbs1)):
            mean_dbs1.append(np.mean(data_dbs1[i]))

        # standard error
        error_dbs1 = []

        for i in range(len(data_dbs1)):
            error_dbs1.append(np.std(data_dbs1[i]))

        ############################################ histo settings #####################################################

        # sessions
        x = 1

        # bar width
        width = 0.5

        colors = [lighter_darkblue]  # [(0.2, 0.2, 1.0, 0.7)]

        # bar positions
        positions = [
            x + 6 * width,
            x + 4.5 * width,
            x + 3 * width,
            x + 1.5 * width,
            x,
            x - 1.5 * width,
            x - 3 * width,
            x - 4.5 * width,
            x - 6 * width,
        ]

        label_y = [
            "$\\mathbf{Cor_{in}}$",
            "$\\mathbf{StrD1}$",
            "$\\mathbf{StrD2}$",
            "$\\mathbf{STN}$",
            "$\\mathbf{GPi}$",
            "$\\mathbf{GPe}$",
            "$\\mathbf{Thalamus}$",
            "$\\mathbf{Cor_{dec}}$",
            "$\\mathbf{StrThal}$",
        ]

        # plot size
        fig, axs = plt.subplots(figsize=(2.5, 2.5))

        xlim_neg = 0
        xlim_pos = 1.1

        ################################################## plot bars #################################################

        ####################### dbs-off data ################################
        for i in range(len(mean_dbs1)):
            # plot means
            axs.barh(
                positions[i],
                mean_dbs1[i],
                height=width,
                color=colors,
            )

            # plot legend
            # axs[0].legend(labels, fontsize="large", loc="upper left")

            for i in range(len(error_dbs1)):
                # plot error bars
                axs.errorbar(
                    x=mean_dbs1[i],
                    y=positions[i],
                    xerr=error_dbs1[i],
                    fmt="none",
                    color="black",
                    capsize=2,
                    capthick=0.5,
                    elinewidth=0.5,
                )

        ############# axis settings ###################
        # horizontal line
        # axs.axvline(x=0, color="black", linestyle="-", linewidth=0.75)

        # y-axis
        axs.set_yticks(positions)
        axs.set_yticklabels(label_y, fontsize=label_size)

        # x-axis
        axs.set_xlabel("average rate", fontweight="bold", fontsize=label_size)
        # axs.set_xticklabels([])
        axs.set_xlim(xlim_neg, xlim_pos)

        axs.tick_params(axis="both", labelsize=label_size)

        # Adjust layout
        plt.tight_layout(
            pad=0,
            h_pad=1.08,
            w_pad=1.08,
            rect=[-0.004, -0.006, 1, 1],
        )

        # save fig
        if session == 0:
            plt.savefig("fig/__fig_activity_change_dbs_off_init__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_off_init__.pdf", format="pdf", dpi=300
            )
        else:
            plt.savefig("fig/__fig_activity_change_dbs_off_learn__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_off_learn__.pdf", format="pdf", dpi=300
            )

        plt.close("all")


#################################################################################################################
############################################ __fig_gpi_scatter__ ################################################
#################################################################################################################


def gpi_scatter():
    ########################################## init variables ########################################

    data_mean_GPi = []
    data_mean_session3 = []

    #################################################### filepath ##########################################################

    for dbs_state in range(1, 5):
        if dbs_state == 1:
            steps = stat.read_json_data(
                f"data/gpi_scatter_data/1_suppression/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 2:
            steps = stat.read_json_data(
                f"data/gpi_scatter_data/2_efferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 3:
            steps = stat.read_json_data(
                f"data/gpi_scatter_data/3_afferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 4:
            steps = stat.read_json_data(
                f"data/gpi_scatter_data/4_passing_fibres/Param_Shortcut1_DBS_State{dbs_state}.json"
            )

        steps = len(steps[0])

        for step in range(steps):

            if dbs_state == 1:
                filepath1 = f"data/gpi_scatter_data/1_suppression/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/1_suppression/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 2:
                filepath1 = f"data/gpi_scatter_data/2_efferent/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/2_efferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 3:
                filepath1 = f"data/gpi_scatter_data/3_afferent/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/3_afferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 4:
                filepath1 = f"data/gpi_scatter_data/4_passing_fibres/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/4_passing_fibres/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"

            ########################################## load and edit data ################################################

            data_GPi = stat.read_json_data(filepath1)
            data_GPi = np.mean(data_GPi)

            data_session3 = stat.read_json_data(filepath2)
            data_session3 = stat.processing_data_param(data_session3)
            data_session3 = stat.mean_data(data_session3)
            data_session3 = data_session3[2]

            # mean
            data_mean_GPi.append(data_GPi)
            data_mean_session3.append(data_session3)

    ##################################################### scatter settings ################################################

    # size plot
    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot()

    ########################## plot data #########################

    # scatterplot
    plt.scatter(data_mean_GPi, data_mean_session3, s=4)
    plt.xlabel("average rate GPi", fontweight="bold", fontsize=label_size)
    plt.ylabel("rewarded decisions", fontweight="bold", fontsize=label_size)

    # means session3 patient data
    line = plt.axhline(y=23.706, color="red", linestyle="--", linewidth=1.0)

    # Legende
    ax.legend(
        [line],
        ["Patient Data Session 3"],
        loc="upper right",
        fontsize=label_size,  # small
    )

    plt.ylim(14, 35)
    plt.xlim(-0.05, 0.6)
    plt.tick_params(axis="both", labelsize=label_size)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.001, -0.005, 1, 1],
    )

    plt.savefig("fig/__fig_gpi_scatter__.png", dpi=300)
    plt.savefig("fig/__fig_gpi_scatter__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
########################################## __fig_load_simulate__ ################################################
#################################################################################################################


def load_simulate():

    # number of simulations
    number_of_simulations = 100
    passingoff = True

    ################################ load dbs-on data ###########################################
    resultON = []

    for i in range(1, 3):
        # without afferent and passing-fibres and dbs-comb
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_2.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultON.append(result)

    ################################ load "simulation" data #####################################
    resultSim = []

    for i in range(1, 3):
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_3.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultSim.append(result)

    ################################ load "loading" data ########################################
    resultLoad = []

    for i in range(1, 3):
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_4.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultLoad.append(result)

    ################################ load dbs-off data ##########################################
    resultOFF = []

    for i in range(1, 3):
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_5.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultOFF.append(result)

    ###################################### edit data ############################################

    index = 2

    result = []
    for i in range(index):
        result.append([resultOFF[i], resultSim[i], resultLoad[i], resultON[i]])

    ################################## boxplot settings #########################################

    if passingoff:
        # Einstellung
        colors = [
            (1, 0.9, 0.9, 0.7),  # very bright red
            (1, 0.6, 0.6, 0.7),  # bright red
            # (0.8, 0, 0, 0.7),  # darkred
        ]
    else:
        colors = [
            (1, 0.9, 0.9, 0.7),  # very bright red
            (1, 0.6, 0.6, 0.7),  # bright red
            (1, 0.4, 0.4, 0.7),  # red
            (0.8, 0, 0, 0.7),  # darkred
        ]

    # bar width
    width = 0.1

    # legend
    if passingoff:
        legend = ["suppression", "efferent"]  # , "dbs-comb"]
    else:
        legend = ["suppression", "efferent", "passing-fibres", "dbs-comb"]

    # plot size
    fig, ax = plt.subplots(figsize=(5, 3))

    # sessions
    session = ["off\noff", "on\noff", "off\non", "on\non"]

    # bar positions
    x = np.arange(len(session))

    if passingoff:
        distance = 0.7
    else:
        distance = 2.25

    for i in range(index):
        positions = x - distance * width

        # Plot boxplots
        ax.boxplot(
            result[i],
            positions=positions,
            widths=width,
            boxprops=dict(facecolor=colors[i]),
            medianprops=dict(color="black"),
            patch_artist=True,
            flierprops=dict(marker="o", color="black", markersize=2),
        )

        distance -= 1.4

    # settings x-axis
    plt.xticks(x, session)
    plt.text(
        -0.8,
        -4.0,
        "acute\n  history",
        fontsize=label_size,
        ha="center",
        va="center",
    )

    plt.tick_params(axis="both", labelsize=label_size)

    ################################################### significance * #############################################

    # function for significance over the boxplots
    def add_star_up(ax, x1, x2, y):
        """Add asterisks for significant differences."""
        y_offset = 0.5
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + y_offset, y + y_offset, y],
            color="black",
            linewidth=1,
        )
        ax.text((x1 + x2) / 2, y + 0.1, "*", fontsize=10, ha="center")

    # function for significance among the boxplots
    def add_star_down(ax, x1, x2, y):
        """Add asterisks for significant differences."""
        y_offset = -0.5
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + y_offset, y + y_offset, y],
            color="black",
            linewidth=1,
        )
        ax.text((x1 + x2) / 2, y - 2.9, "*", fontsize=10, ha="center")

    add_star_up(ax, 0.93, 2.93, 38)
    add_star_up(ax, 1.07, 3.07, 40)
    add_star_up(ax, -0.07, 1.93, 33)
    add_star_up(ax, 0.07, 2.07, 35)
    add_star_down(ax, 1.93, 2.93, 3)

    # legend
    legend_bars = [
        plt.bar(0, 0, color=colors[i], label=legend[i]) for i in range(len(colors))
    ]

    #################################################### plot settings #############################################

    plt.legend(handles=legend_bars, loc="upper left", fontsize=label_size)  # small

    # setting y-axis
    plt.ylabel("unrewarded decisions", fontweight="bold", fontsize=label_size)
    plt.ylim(0, 46)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.006, -0.004, 1, 1],
    )

    plt.savefig("fig/__fig_load_simulate__.png", dpi=300)
    plt.savefig("fig/__fig_load_simulate__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
################################### __appendix__fig_load_simulate_dbscomb__ ######################################
#################################################################################################################


def load_simulate_dbscomb():

    # number of simulations
    number_of_simulations = 100
    passingoff = True

    ################################ load dbs-on data ###########################################
    resultON = []

    for i in range(1, 6):
        # without afferent and passing-fibres and dbs-comb
        if i == 2:
            continue
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_2.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultON.append(result)

    ################################ load "simulation" data #####################################
    resultSim = []

    for i in range(1, 6):
        if i == 2:
            continue
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_3.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultSim.append(result)

    ################################ load "loading" data ########################################
    resultLoad = []

    for i in range(1, 6):
        if i == 2:
            continue
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_4.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultLoad.append(result)

    ################################ load dbs-off data ##########################################
    resultOFF = []

    for i in range(1, 6):
        if i == 2:
            continue
        if i == 3:
            continue
        if i == 4 and passingoff:
            continue

        filepath = f"data/load_simulation_data/load_data/Results_DBS_State_{i}_Condition_5.json"
        result = stat.read_json_data(filepath)
        result = stat.processing_habit_session3(result, number_of_simulations)
        result = result.T
        result = result[0].tolist()
        resultOFF.append(result)

    ###################################### edit data ############################################

    # if passingoff:
    #    index = 3
    # else:
    #    index = 4

    index = 2

    result = []
    for i in range(index):
        result.append([resultOFF[i], resultSim[i], resultLoad[i], resultON[i]])

    ################################## boxplot settings #########################################

    if passingoff:
        # Einstellung
        colors = [
            (1, 0.9, 0.9, 0.7),  # very bright red
            # (1, 0.6, 0.6, 0.7),  # bright red
            (0.8, 0, 0, 0.7),  # darkred
        ]
    else:
        colors = [
            (1, 0.9, 0.9, 0.7),  # very bright red
            (1, 0.6, 0.6, 0.7),  # bright red
            (1, 0.4, 0.4, 0.7),  # red
            (0.8, 0, 0, 0.7),  # darkred
        ]

    # bar width
    width = 0.1

    # legend
    if passingoff:
        legend = ["suppression", "dbs-comb"]
    else:
        legend = ["suppression", "efferent", "passing-fibres", "dbs-comb"]

    # plot size
    fig, ax = plt.subplots(figsize=(5, 3))

    # sessions
    session = ["off\noff", "on\noff", "off\non", "on\non"]

    # bar positions
    x = np.arange(len(session))

    if passingoff:
        distance = 0.7
    else:
        distance = 2.25

    for i in range(index):
        positions = x - distance * width

        # Plot boxplots
        ax.boxplot(
            result[i],
            positions=positions,
            widths=width,
            boxprops=dict(facecolor=colors[i]),
            medianprops=dict(color="black"),
            patch_artist=True,
            flierprops=dict(marker="o", color="black", markersize=2),
        )

        distance -= 1.4

    # settings x-axis
    plt.xticks(x, session)
    plt.text(
        -0.8,
        -4.0,
        "acute\n  history",
        fontsize=label_size,
        ha="center",
        va="center",
    )

    # legend
    legend_bars = [
        plt.bar(0, 0, color=colors[i], label=legend[i]) for i in range(len(colors))
    ]

    plt.legend(handles=legend_bars, loc="upper left", fontsize=label_size)  # small

    # setting y-axis
    plt.ylabel("unrewarded decisions", fontweight="bold", fontsize=label_size)
    plt.ylim(0, 46)

    plt.tick_params(axis="both", labelsize=label_size)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.006, -0.004, 1, 1],
    )

    plt.savefig("fig/__fig_load_simulate_dbscomb__.png", dpi=300)
    plt.savefig("fig/__fig_load_simulate_dbscomb__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
############################################ __fig_dbs_parameter__ ##############################################
#################################################################################################################


def dbs_parameter():
    ########################################## init variables ########################################

    data_suppression = []
    data_efferent = []
    data_afferent = []
    data_passing_fibres = []

    dbs_states = [
        "suppression",
        "efferent",
        "afferent",
        "passing fibres GPe-STN",
    ]

    parameter_name = [
        r"$\mathbf{\alpha_{suppress}}$",
        r"$\mathbf{\alpha_{axon}}$",
        r"$\mathbf{\alpha_{axon}}$",
        r"$\mathbf{\alpha_{axon}}$",
    ]

    line_color = ["darkblue", "darkred", "orange"]

    ######################################################## load data ##########################################################

    for dbs_state in range(1, 5):
        if dbs_state == 1:
            steps = stat.read_json_data(
                f"data/parameter_data/1_suppression/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 2:
            steps = stat.read_json_data(
                f"data/parameter_data/2_efferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 3:
            steps = stat.read_json_data(
                f"data/parameter_data/3_afferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 4:
            steps = stat.read_json_data(
                f"data/parameter_data/4_passing_fibres/Param_Shortcut1_DBS_State{dbs_state}.json"
            )

        steps = len(steps[0])

        for step in range(steps):

            if dbs_state == 1:
                filepath = f"data/parameter_data/1_suppression/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/parameter_data/1_suppression/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 2:
                filepath = f"data/parameter_data/2_efferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/parameter_data/2_efferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 3:
                filepath = f"data/parameter_data/3_afferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/parameter_data/3_afferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 4:
                filepath = f"data/parameter_data/4_passing_fibres/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/parameter_data/4_passing_fibres/Param_Shortcut1_DBS_State{dbs_state}.json"

            ####################################### load and edit data ##########################################

            data = stat.read_json_data(filepath)
            data = stat.processing_data_param(data)
            data = stat.mean_data(data)
            data = stat.processing_performance_param(data)

            if step == 0:
                param = stat.read_json_data(filepath_param)
                param = stat.processing_data_param(param)
                param = np.array(param)
                param = param.T

            if dbs_state == 1:
                data_suppression.append(data)
                if step == 0:
                    param_suppression = param[0]

            if dbs_state == 2:
                data_efferent.append(data)
                if step == 0:
                    param_efferent = param[0]

            if dbs_state == 3:
                data_afferent.append(data)
                if step == 0:
                    param_afferent = param[0]

            if dbs_state == 4:
                data_passing_fibres.append(data)
                if step == 0:
                    param_passing_fibres = (
                        param[0] * 0.05
                    )  # param_passing_fibres = fibre_strength * axon rate amp

    data_suppression = np.array(data_suppression)
    data_suppression = data_suppression.T
    data_efferent = np.array(data_efferent)
    data_efferent = data_efferent.T
    data_afferent = np.array(data_afferent)
    data_afferent = data_afferent.T
    data_passing_fibres = np.array(data_passing_fibres)
    data_passing_fibres = data_passing_fibres.T

    ################################################### plot settings ##########################################

    # Fenster für Diagramme erstellen
    fig = plt.figure(figsize=(5, 4.5))
    idx = 1

    parameter_lines = [0.1, 0.05, 0.05, 7.5 * 0.05]

    # Diagramme für jede Population erstellen
    for i in dbs_states:
        # Diagramm hinzufügen
        ax = fig.add_subplot(2, 2, idx)

        # Abstand zwischen Diagrammen
        plt.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4
        )

        ########################## select data ##########################

        if i == "suppression":
            data = data_suppression
            param = param_suppression
        if i == "efferent":
            data = data_efferent
            param = param_efferent
        if i == "afferent":
            data = data_afferent
            param = param_afferent
        if i == "passing fibres GPe-STN":
            data = data_passing_fibres
            param = param_passing_fibres

        ############################ plot data #########################

        for j in range(3):
            ax.plot(
                param,
                data[j],
                label=f"Session {j + 1}",
                linewidth=1,
                color=line_color[j],
            )

        ax.axvline(
            x=parameter_lines[idx - 1], color="black", linestyle="-", linewidth=1
        )

        # plot border at 1.0
        tresh = []

        for k in range(len(param)):
            tresh.append(1.0)
        ax.plot(
            param,
            tresh,
            linewidth=1,
            color="red",
            linestyle="--",
            label="Patient Data",
        )
        # if i == "efferent":
        # ax.legend(fontsize=label_size)

        # Achsen beschriften
        ax.set_xlabel(parameter_name[idx - 1])
        ax.set_title(i, fontweight="bold", fontsize=label_size)
        if i == "suppression" or i == "afferent":
            ax.set_ylabel(
                "normalized\nrewarded decisions", fontweight="bold", fontsize=label_size
            )
        else:
            ax.set_yticklabels([])

        # scale x-axis
        if idx == 1 and idx == 2:
            ax.set_xticks(np.arange(0, 0.70, 0.2))
        if idx == 3:
            ax.set_xticks(np.arange(0, 0.31, 0.1))
        if idx == 4:
            ax.set_xticks(np.arange(0, 1.60, 0.5))

        ax.set_xlim(min(param), max(param))
        ax.tick_params(axis="both", which="major", labelsize=label_size)

        idx += 1
        plt.ylim(0.5, 1.4)

    ###################### legend ##########################

    # global legend below the subplots centered in the figure
    handles, labels = ax.get_legend_handles_labels()
    fig = plt.gcf()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        fontsize=label_size,
        borderaxespad=0,
        title_fontproperties={"weight": "bold", "size": label_size},
    )

    # get the coordinates of the legend box
    legend_bbox = legend.get_window_extent()
    legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.005, legend_bbox.y1, 1, 1],
    )

    ################# save fig ######################

    plt.savefig("fig/__fig_dbs_parameter__.png", dpi=300)
    plt.savefig("fig/__fig_dbs_parameter__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
################################### __appendix_fig_parameter_gpi_inhib__ ########################################
#################################################################################################################


def parameter_gpi_inhib():
    ##################################################  init varables ######################################################

    data_GPi_suppression = []
    data_GPi_efferent = []
    data_GPi_afferent = []
    data_GPi_passing_fibres = []

    data_suppression = []
    data_efferent = []
    data_afferent = []
    data_passing_fibres = []

    dbs_states = [
        "suppression",
        "efferent",
        "afferent",
        "passing fibres GPe-STN",
    ]

    parameter_name = [
        r"$\mathbf{\alpha_{suppress}}$",
        r"$\mathbf{\alpha_{axon}}$",
        r"$\mathbf{\alpha_{axon}}$",
        r"$\mathbf{\alpha_{axon}}$",
    ]

    ######################################################## filepath ##########################################################

    for dbs_state in range(1, 5):
        if dbs_state == 1:
            steps = stat.read_json_data(
                f"data/parameter_data/1_suppression/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 2:
            steps = stat.read_json_data(
                f"data/parameter_data/2_efferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 3:
            steps = stat.read_json_data(
                f"data/parameter_data/3_afferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            )
        if dbs_state == 4:
            steps = stat.read_json_data(
                f"data/parameter_data/4_passing_fibres/Param_Shortcut1_DBS_State{dbs_state}.json"
            )

        steps = len(steps[0])

        for step in range(steps):

            if dbs_state == 1:
                filepath1 = f"data/gpi_scatter_data/1_suppression/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/1_suppression/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/gpi_scatter_data/1_suppression/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 2:
                filepath1 = f"data/gpi_scatter_data/2_efferent/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/2_efferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/gpi_scatter_data/2_efferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 3:
                filepath1 = f"data/gpi_scatter_data/3_afferent/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/3_afferent/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/gpi_scatter_data/3_afferent/Param_Shortcut1_DBS_State{dbs_state}.json"
            if dbs_state == 4:
                filepath1 = f"data/gpi_scatter_data/4_passing_fibres/mean_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath2 = f"data/parameter_data/4_passing_fibres/Results_Shortcut1_DBS_State{dbs_state}_Step{step}.json"
                filepath_param = f"data/gpi_scatter_data/4_passing_fibres/Param_Shortcut1_DBS_State{dbs_state}.json"

            ############################################### load and edit data ###############################################

            data_GPi = stat.read_json_data(filepath1)
            data_GPi = np.mean(data_GPi)

            data = stat.read_json_data(filepath2)
            data = stat.processing_data_param(data)
            data = stat.mean_data(data)
            data = stat.processing_performance_param(data)

            if step == 0:
                param = stat.read_json_data(filepath_param)
                param = stat.processing_data_param(param)
                param = np.array(param)
                param = param.T

            if dbs_state == 1:
                data_GPi_suppression.append(data_GPi)
                data_suppression.append(data)
                if step == 0:
                    param_suppression = param[0]

            if dbs_state == 2:
                data_GPi_efferent.append(data_GPi)
                data_efferent.append(data)
                if step == 0:
                    param_efferent = param[0]

            if dbs_state == 3:
                data_GPi_afferent.append(data_GPi)
                data_afferent.append(data)
                if step == 0:
                    param_afferent = param[0]

            if dbs_state == 4:
                data_GPi_passing_fibres.append(data_GPi)
                data_passing_fibres.append(data)
                if step == 0:
                    param_passing_fibres = param[0] * 0.05

    data_suppression = np.array(data_suppression)
    data_suppression = np.abs(data_suppression)
    mean_suppression = np.mean(data_suppression, axis=1)
    data_efferent = np.array(data_efferent)
    data_efferent = np.abs(data_efferent)
    mean_efferent = np.mean(data_efferent, axis=1)
    data_afferent = np.array(data_afferent)
    data_afferent = np.abs(data_afferent)
    mean_afferent = np.mean(data_afferent, axis=1)
    data_passing_fibres = np.array(data_passing_fibres)
    data_passing_fibres = np.abs(data_passing_fibres)
    mean_passing_fibres = np.mean(data_passing_fibres, axis=1)

    ################################################### plot settings ##########################################

    # plot size
    fig = plt.figure(figsize=(5, 4.5))
    idx = 1

    for i in dbs_states:
        ax1 = fig.add_subplot(2, 2, idx)

        # distance between plots
        plt.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4
        )

        label = [
            "Mean Sessions",
            "Mean Rate GPi",
            "Patient Data",
        ]
        color = ["orange", "darkblue"]

        ########################### select data ##########################

        if i == "suppression":
            data = [mean_suppression, data_GPi_suppression]
            param = param_suppression
        if i == "efferent":
            data = [mean_efferent, data_GPi_efferent]
            param = param_efferent
        if i == "afferent":
            data = [mean_afferent, data_GPi_afferent]
            param = param_afferent
        if i == "passing fibres GPe-STN":
            data = [mean_passing_fibres, data_GPi_passing_fibres]
            param = param_passing_fibres

        ############################## plot data #########################

        (line1,) = ax1.plot(
            param,
            data[0],
            label=label[0],
            linewidth=1,
            color=color[0],
        )
        # settings axis
        ax1.tick_params(axis="both", which="major", labelsize=label_size)
        ax1.set_xlabel(parameter_name[idx - 1])
        ax1.set_title(i, fontweight="bold", fontsize=label_size)
        if i == "suppression" or i == "afferent":
            ax1.set_ylabel(
                "normalized\nrewarded decisions", fontweight="bold", fontsize=label_size
            )
        else:
            ax1.set_yticklabels([])

        # border at 1.0
        tresh = []

        for k in range(len(param)):
            tresh.append(1.0)
        (line3,) = ax1.plot(
            param,
            tresh,
            linewidth=1,
            color="red",
            linestyle="--",
            label=label[2],
        )

        # average rate GPi
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            param,
            data[1],
            label=label[1],
            linewidth=1,
            color=color[1],
        )
        # settings axis
        ax2.tick_params(axis="both", which="major", labelsize=label_size)
        ax2.set_xlabel(parameter_name[idx - 1])
        ax2.set_title(i, fontweight="bold", fontsize=label_size)
        if i == "efferent" or i == "passing fibres GPe-STN":
            ax2.set_ylabel("average rate", fontweight="bold", fontsize=label_size)
        else:
            ax2.set_yticklabels([])

        # scale x-axis
        if idx == 1:
            ax1.set_xticks(np.arange(0, 0.70, 0.2))
        if idx == 2:
            ax2.set_xticks(np.arange(0, 0.70, 0.2))
        if idx == 3:
            ax1.set_xticks(np.arange(0, 0.31, 0.1))
        if idx == 4:
            ax2.set_xticks(np.arange(0, 1.60, 0.5))

        # min/max x-/y-axis
        ax1.set_ylim(0.5, 1.4)
        ax2.set_ylim(-0.1, 0.7)
        ax1.set_xlim(min(param), max(param))
        ax2.set_xlim(min(param), max(param))

        idx += 1

    plt.tick_params(axis="both", labelsize=label_size)

    ###################### legend ##########################

    # global legend below the subplots centered in the figure
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig = plt.gcf()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        fontsize=label_size,
        borderaxespad=0,
        title_fontproperties={"weight": "bold", "size": label_size},
    )

    # get the coordinates of the legend box
    legend_bbox = legend.get_window_extent()
    legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[-0.004, legend_bbox.y1, 1.002, 1],
    )

    ################## save fig #########################

    plt.savefig("fig/__appendix_fig_parameter_gpi_inhib__.png", dpi=300)
    plt.savefig("fig/__appendix_fig_parameter_gpi_inhib__.pdf", format="pdf", dpi=300)
    plt.close("all")


#################################################################################################################
################################### __fig_weights_over_time__ ########################################
#################################################################################################################


def get_w_direct(loaded_variables, dbs_state, sim_id, trial):
    w1 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "ITStrD1"
        ][trial]
    )
    w2 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "StrD1GPi"
        ][trial]
    )
    return np.transpose(np.matmul(w1, w2))


def get_w_indirect(loaded_variables, dbs_state, sim_id, trial):
    w1 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "ITStrD2"
        ][trial]
    )
    w2 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "StrD2GPe"
        ][trial]
    )
    return np.transpose(np.matmul(w1, w2))


def get_w_hyperdirect(loaded_variables, dbs_state, sim_id, trial):
    w1 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "ITSTN"
        ][trial]
    )
    w2 = np.transpose(
        loaded_variables[f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"][
            "STNGPi"
        ][trial]
    )
    return np.transpose(np.matmul(w1, w2))


def weights_over_time_boxplots(
    df,
    specific_dbs_types=["suppression", "efferent", "dbs-comb", "OFF"],
    specific_sessions=[3],
):
    df = df.copy()
    # Remove all rows where dbs_state is OFF and dbs_type is not dbs-comb (i.e. only
    # keep OFF for dbs_type dbs-comb)
    df = df[~((df["dbs_state"] == "OFF") & (df["dbs_type"] != "dbs-comb"))]

    # For the rows in which dbs_state is OFF change the dbs_type to OFF
    df.loc[df["dbs_state"] == "OFF", "dbs_type"] = "OFF"

    # Remove the dbs_state column
    df = df.drop(columns=["dbs_state"])

    # Define the specific dbs_type values to plot
    save_string_dbs_types = "".join([s[0] for s in specific_dbs_types])

    # Define the sessions to plot
    save_string_sessions = "".join([str(s) for s in specific_sessions])

    # Filter the DataFrame to include only the specified dbs_type values
    df_filtered = df[df["dbs_type"].isin(specific_dbs_types)]

    # Filter the DataFrame to include only the specified sessions
    df_filtered = df_filtered[df_filtered["session"].isin(specific_sessions)]

    # Create subplots for each weight type
    weight_types = [
        ("direct", 0),
        ("direct", 1),
        ("indirect", 0),
        ("indirect", 1),
        ("hyperdirect", 0),
        ("hyperdirect", 1),
        ("shortcut", 0),
        ("shortcut", 1),
        ("dopa_predict", 0),
    ]

    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    # Create a boxplot for each weight type
    for i, weight_type in enumerate(weight_types):
        x = "session" if len(specific_sessions) > 1 else "dbs_type"
        hue = "dbs_type"
        hue_order = specific_dbs_types
        order = specific_sessions if len(specific_sessions) > 1 else specific_dbs_types
        # filter the data for the specific weight type
        df_filtered_plot = df_filtered[df_filtered["pathway"] == weight_type[0]]
        df_filtered_plot = df_filtered_plot[
            df_filtered_plot["channel"] == weight_type[1]
        ]
        # average over trials
        df_filtered_plot = df_filtered_plot.groupby(
            ["sim_id", "dbs_type", "session", "pathway", "channel"], as_index=False
        ).mean()
        ax = sns.boxplot(
            x=x,
            y="w",
            hue=hue,
            palette={
                "suppression": (1, 0.7, 0.7, 0.8),
                "efferent": (1, 0.5, 0.5, 0.8),
                "dbs-comb": (0.8, 0, 0, 0.8),
                "OFF": "darkblue",
                "passing": "yellow",
                "afferent": "green",
            },
            data=df_filtered_plot,
            ax=axes[i],
            showmeans=True,
            meanprops={
                "markerfacecolor": "black",
                "markeredgecolor": "white",
            },
            linecolor="black",
            order=order,
            hue_order=hue_order,
        )
        axes[i].set_title(weight_type)
        if len(specific_sessions) > 1:
            axes[i].legend_.remove()  # Remove individual legends

        # # make pariwise tests and annotate
        # if len(specific_sessions) == 1:
        #     # pairs for statistic comparison
        #     pairs = [
        #         ("suppression", "OFF"),
        #         ("efferent", "OFF"),
        #         ("dbs-comb", "OFF"),
        #     ]

        #     # make pairwise tests
        #     a = Annotator(ax, pairs=pairs)
        #     a.configure(test="t-test")
        #     a.apply_test()
        #     a.annotate()  # TODO find out how annotater works, make two-ways repeated anovas for dbs_state and dbs_type for each weight, e.g. w_direct_0 and than make paired t-tests with pingouin, then annotate p results in boxplot

    if len(specific_sessions) > 1:
        # Create a single legend outside the subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(specific_dbs_types))

    # Adjust layout
    if len(specific_sessions) > 1:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    plt.savefig(
        f"fig/__fig_weights_boxplots_{save_string_dbs_types}_{save_string_sessions}__.png",
        dpi=300,
    )
    plt.close("all")


def get_weights_over_time_data_frame(
    n_sims, w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict
):
    # create a pandas dataframe with columns "w", "sim_id", "dbs_type" (suppression,
    # efferent, afferent, passing, dbs-comb), "dbs_state" (ON, OFF), "session" (1,2,3),
    # "pathway" (direct, indirect, hyperdirect, shortcut, dopa_predict), "channel" (0, 1)
    # and "trial" (0-119)
    df = {
        "w": [],
        "sim_id": [],
        "dbs_type": [],
        "dbs_state": [],
        "trial": [],
        "session": [],
        "pathway": [],
        "channel": [],
    }

    weight_arrays_dict = {
        "w_direct": w_direct,
        "w_indirect": w_indirect,
        "w_hyperdirect": w_hyperdirect,
        "w_shortcut": w_shortcut,
        "w_dopa_predict": w_dopa_predict,
    }

    for sim_id in range(n_sims):
        for dbs_state in ["ON", "OFF"]:
            for dbs_type in [
                "suppression",
                "efferent",
                "dbs-comb",
                "afferent",
                "passing",
            ]:
                for session in [1, 2, 3]:
                    if dbs_state == "OFF":
                        dbs_idx = 0
                    else:
                        dbs_idx = {
                            "suppression": 1,
                            "efferent": 2,
                            "afferent": 3,
                            "passing": 4,
                            "dbs-comb": 5,
                        }[dbs_type]
                    # each session has 40 trials
                    trial_idx_start = (session - 1) * 40
                    trial_idx_end = session * 40
                    # loop over trials
                    for trial in range(trial_idx_start, trial_idx_end):
                        # loop over pathways
                        for pathway in [
                            "direct",
                            "indirect",
                            "hyperdirect",
                            "shortcut",
                            "dopa_predict",
                        ]:
                            for pathway_idx in [0, 1]:
                                # for dopa_predict skip channel 1
                                if pathway == "dopa_predict" and pathway_idx == 1:
                                    continue
                                df["w"].append(
                                    weight_arrays_dict[f"w_{pathway}"][
                                        sim_id,
                                        dbs_idx,
                                        trial,
                                        pathway_idx,
                                        0,
                                    ]
                                )
                                df["sim_id"].append(sim_id)
                                df["dbs_type"].append(dbs_type)
                                df["dbs_state"].append(dbs_state)
                                df["session"].append(session)
                                df["pathway"].append(pathway)
                                df["channel"].append(pathway_idx)
                                df["trial"].append(trial)

    return pd.DataFrame(df)


def get_weights_over_time_arrays(n_sims, n_dbs, n_trials):
    # we have weights for each sim (0-99), shortcut (0-fixed,1-plastic), dbs state
    # (0-DBS-OFF,1-suppression,2-efferent,3-afferent,4-passing,5-dbs-comb)

    name_list = []
    for sim_id in range(n_sims):
        for dbs_state in range(n_dbs):
            name_list.append(
                f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"
            )
    loaded_variables = load_variables(name_list=name_list, path="data/simulation_data/")

    # in each loaded file we have these weight arrays:
    # ITThal
    # (120, 2, 2)
    # ITStrD1
    # (120, 4, 2)
    # ITStrD2
    # (120, 4, 2)
    # ITSTN
    # (120, 4, 2)
    # StrD1SNc
    # (120, 1, 4)
    # StrD1GPi
    # (120, 2, 4)
    # STNGPi
    # (120, 2, 4)
    # StrD2GPe
    # (120, 2, 4)

    # get the combined matrices for all time points / trials
    # the last two dimensions are (post, pre)
    w_direct = np.zeros((n_sims, n_dbs, n_trials, 2, 2))
    w_indirect = np.zeros((n_sims, n_dbs, n_trials, 2, 2))
    w_hyperdirect = np.zeros((n_sims, n_dbs, n_trials, 2, 2))
    w_shortcut = np.zeros((n_sims, n_dbs, n_trials, 2, 2))
    w_dopa_predict = np.zeros((n_sims, n_dbs, n_trials, 1, 4))

    # - matrix multiply
    #   - ITStrD1 * StrD1GPi --> w_direct
    #   - ITStrD2 * StrD2GPe --> w_indirect
    #   - ITSTN * STNGPi --> w_hyperdirect
    # - sort others
    #   - ITThal --> w_shortcut
    #   - StrD1SNc --> w_dopa_predict
    for sim_id in range(n_sims):
        for dbs_state in range(n_dbs):
            for trial in range(n_trials):
                w_direct[sim_id, dbs_state, trial] = get_w_direct(
                    loaded_variables, dbs_state, sim_id, trial
                )
                w_indirect[sim_id, dbs_state, trial] = get_w_indirect(
                    loaded_variables, dbs_state, sim_id, trial
                )
                w_hyperdirect[sim_id, dbs_state, trial] = get_w_hyperdirect(
                    loaded_variables, dbs_state, sim_id, trial
                )
                w_shortcut[sim_id, dbs_state, trial] = loaded_variables[
                    f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"
                ]["ITThal"][trial]

                w_dopa_predict[sim_id, dbs_state, trial] = loaded_variables[
                    f"plastic_weights_Shortcut1_DBS_State{dbs_state}_sim{sim_id}"
                ]["StrD1SNc"][trial]

    # - average over indizes/neurons:
    #   - w_direct, w_indirect, w_hyperdirect, w_shortcut --> over IT dimension
    #   - w_dopa_predict --> over StrD1 dimension
    # - after mean the shapes are (n_sims, n_dbs, n_trials, 1, 1) for dopa_predict
    #   and (n_sims, n_dbs, n_trials, 2, 1) for the others
    w_direct = np.mean(w_direct, axis=4, keepdims=True)  # IT is pre
    w_indirect = np.mean(w_indirect, axis=4, keepdims=True)  # IT is pre
    w_hyperdirect = np.mean(w_hyperdirect, axis=4, keepdims=True)  # IT is pre
    w_shortcut = np.mean(w_shortcut, axis=4, keepdims=True)  # IT is pre
    w_dopa_predict = np.mean(w_dopa_predict, axis=4, keepdims=True)  # StrD1 is pre

    return w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict


def weights_over_time_lineplots_old(
    w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict, n_dbs
):
    # for plotting weights through time average over sim_ids (the reversal order is the
    # same for all sims)
    # shape (n_dbs, n_trials, 2, 1) after averaging over sim_ids
    w_direct = np.mean(w_direct, axis=0)
    w_indirect = np.mean(w_indirect, axis=0)
    w_hyperdirect = np.mean(w_hyperdirect, axis=0)
    w_shortcut = np.mean(w_shortcut, axis=0)
    # shape (n_dbs, n_trials, 1, 1) after averaging over sim_ids
    w_dopa_predict = np.mean(w_dopa_predict, axis=0)

    arrays = [w_direct, w_indirect, w_hyperdirect, w_shortcut]

    # Subplot titles for each row and column
    subplot_titles = [
        ["channel 0", "channel 1"],
        ["", ""],
        ["", ""],
        ["", ""],
    ]
    subplot_ylabels = [
        ["direct", ""],
        ["indirect", ""],
        ["hyperdirect", ""],
        ["shortcut", ""],
    ]
    dbs_state_names = [
        ["DBS-OFF", True],
        ["suppression", True],
        ["efferent", True],
        ["afferent", False],
        ["passing fibres GPe-STN", False],
        ["dbs-comb", True],
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 8))

    # Plot each array on its respective subplot, splitting by the first dimension
    for row_idx, (row_axes, title_row, ylabel_row) in enumerate(
        zip(axes, subplot_titles, subplot_ylabels)
    ):
        # get ylim for the row
        ylim = (
            arrays[row_idx][
                np.array([dbs_state_names[dbs][1] for dbs in range(n_dbs)]), :, :, 0
            ].min(),
            arrays[row_idx][
                np.array([dbs_state_names[dbs][1] for dbs in range(n_dbs)]), :, :, 0
            ].max(),
        )
        # loop over columns/post neurons i.e. "channels"
        for column_idx, (ax, title, ylabel) in enumerate(
            zip(row_axes, title_row, ylabel_row)
        ):
            # set ylim for the column
            ax.set_ylim(ylim)
            # activate grid
            ax.grid(True)
            # plot data
            for dbs in range(n_dbs):
                if dbs_state_names[dbs][1]:
                    ax.plot(
                        arrays[row_idx][dbs, :, column_idx, 0],
                        label=dbs_state_names[dbs][0],
                    )
            # Set title and labels
            if title != "":
                ax.set_title(title)
            if ylabel != "":
                ax.set_ylabel(ylabel)
            # Setting y-axis tick format to two decimal places
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
            # for last subplot add legend
            if row_idx * 2 + column_idx == 7:
                ax.legend()

    # Set common labels and layout
    # fig.suptitle("Title", fontsize=16)
    fig.tight_layout(
        # rect=[0, 0.03, 1, 0.95],
    )

    plt.savefig("fig/__fig_weights_over_time_old__.png", dpi=300)
    plt.close("all")


def weights_over_time_lineplots(data):
    font_size = 9

    # exclude the pathway dopa_predict
    data = data[data["pathway"] != "dopa_predict"]

    save_time_for_formatting = False
    if save_time_for_formatting:
        # only include pathway "shortcut" and "hyperdirect"
        data = data[data["pathway"].isin(["shortcut", "hyperdirect"])]

        # only include dbs_types "suppression", "efferent"
        data = data[data["dbs_type"].isin(["suppression", "efferent"])]

    # Get unique pathways and dbs_types
    pathways = data["pathway"].unique()
    dbs_types = data["dbs_type"].unique()

    # Create subplots
    n = len(pathways)
    m = len(dbs_types)
    fig, axes = plt.subplots(
        n, m, figsize=(16.5 / 2.54, 4 * n), sharex=True, sharey=True
    )

    # If there's only one row or column, axes is not a 2D array
    if n == 1 or m == 1:
        axes = axes.reshape(n, m)

    # define height of significance annotation
    significance_height = 0.045

    # Function to perform t-test
    def perform_ttest(subset, channel):
        p_val_list = []
        # filter data for session 3
        subset = subset[subset["session"] == 3]
        for trial in subset["trial"].unique():
            trial_data = subset[
                (subset["trial"] == trial) & (subset["channel"] == channel)
            ]
            on_values = trial_data[trial_data["dbs_state"] == "ON"]["w"]
            off_values = trial_data[trial_data["dbs_state"] == "OFF"]["w"]
            if len(on_values) > 1 and len(off_values) > 1:
                results = pg.ttest(on_values, off_values, paired=True)
                p_val_list.append([trial, results["p-val"].values[0]])
            else:
                p_val_list.append([trial, None])
        p_val_array = np.array(p_val_list)
        # correct the p-values for multiple comparisons
        reject = multipletests(p_val_array[:, 1], alpha=0.05, method="fdr_bh")[0]
        p_val_array = np.column_stack((p_val_array, reject))
        return p_val_array

    # Loop through pathways and dbs_types to create each subplot
    for i, pathway in enumerate(pathways):
        for j, dbs_type in enumerate(dbs_types):
            ax = axes[i, j]

            # Filter the data for the current pathway and dbs_type and plot the lineplot
            subset = data[(data["pathway"] == pathway) & (data["dbs_type"] == dbs_type)]
            sns.lineplot(
                data=subset, x="trial", y="w", hue="dbs_state", style="channel", ax=ax
            )

            # horizontal line at 0, vertical lines at 40 and 80
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(40, color="black", linewidth=0.5)
            ax.axvline(80, color="black", linewidth=0.5)

            # Perform t-tests for each channel
            channels = subset["channel"].unique()
            for channel in channels:
                ttest_results = perform_ttest(subset, channel)
                for trial, p_val, reject in ttest_results:
                    if reject:
                        # Annotate significance using fill_between below y==0
                        ax.fill_between(
                            x=[trial - 0.5, trial + 0.5],
                            y1=-significance_height * channel,
                            y2=-significance_height * channel - significance_height,
                            color="black",
                            alpha=1 - 0.5 * channel,
                            # without line
                            linewidth=0,
                        )

            # Set titles, labels, etc.
            # title only in the first row
            if i == 0:
                ax.set_title(dbs_type, fontsize=font_size, fontweight="bold")
            else:
                ax.set_title("")
            ax.set_xlabel("Trial", fontsize=font_size, fontweight="bold")
            ax.set_ylabel(f"w ({pathway})", fontsize=font_size, fontweight="bold")

            # no legends within subplots
            ax.get_legend().remove()

            # set x limit to 0-119
            ax.set_xlim(0, 119)

            # set tick parameters
            ax.tick_params(axis="both", which="major", labelsize=font_size)

    # global legend below the subplots centered in the figure
    handles, labels = ax.get_legend_handles_labels()
    fig = plt.gcf()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        fontsize=font_size,
        borderaxespad=0,
        title_fontproperties={"weight": "bold", "size": font_size},
    )

    # Access the legend text objects
    legend_texts = legend.get_texts()

    # Define column headings (index them based on your legend layout)
    column_headings_dict = {
        legend_text.get_text(): legend_text
        for legend_text in legend_texts
        if legend_text.get_text() in ["dbs_state", "channel"]
    }
    print(legend_texts)
    print(column_headings_dict)
    heading_new_text_dict = {
        "dbs_state": "DBS",
        "channel": "Channel",
    }

    # Make column headings bold
    for heading_text, heading in column_headings_dict.items():
        heading.set_fontweight("bold")
        # change text of the column headings
        heading.set_text(heading_new_text_dict[heading_text])

    # get the coordinates of the legend box
    legend_bbox = legend.get_window_extent()
    legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())

    # set the lower y limit to make place for the singificance annotations
    ax.set_ylim(bottom=-2 * significance_height)

    # Adjust layout
    plt.tight_layout(
        pad=0,
        h_pad=1.08,
        w_pad=1.08,
        rect=[0, legend_bbox.y1, 1, 1],
    )

    plt.savefig("fig/__fig_weights_over_time__.png", dpi=300)
    plt.close("all")


def weights_over_time():

    n_sims = 100
    n_dbs = 6
    n_trials = 120

    # get data
    w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict = (
        get_weights_over_time_arrays(n_sims, n_dbs, n_trials)
    )

    df = get_weights_over_time_data_frame(
        n_sims, w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict
    )

    # Plot weights over time as lineplots
    weights_over_time_lineplots_old(
        w_direct, w_indirect, w_hyperdirect, w_shortcut, w_dopa_predict, n_dbs
    )
    weights_over_time_lineplots(df)

    # Create boxplots for all dbs types and sessions
    weights_over_time_boxplots(
        df,
        specific_dbs_types=[
            "suppression",
            "efferent",
            "dbs-comb",
            "afferent",
            "passing",
            "OFF",
        ],
        specific_sessions=[1, 2, 3],
    )
    # Only for specific dbs types
    weights_over_time_boxplots(
        df,
        specific_dbs_types=["suppression", "efferent", "dbs-comb", "OFF"],
        specific_sessions=[1, 2, 3],
    )
    # Only for session 3
    weights_over_time_boxplots(
        df,
        specific_dbs_types=["suppression", "efferent", "dbs-comb", "OFF"],
        specific_sessions=[3],
    )

    return

    # Perform two-way repeated measures ANOVA for each weight variable
    # only for suppression, efferent, dbs-comb and session 3
    specific_dbs_types = ["suppression", "efferent", "dbs-comb"]
    specific_sessions = [3]

    # Filter the DataFrame to include only the specified dbs_type values
    df_filtered = df[df["dbs_type"].isin(specific_dbs_types)]

    # Filter the DataFrame to include only the specified sessions
    df_filtered = df_filtered[df_filtered["session"].isin(specific_sessions)]

    # Perform two-way repeated measures ANOVA for each weight variable
    weight_vars = [
        "w_direct_0",
        "w_direct_1",
        "w_indirect_0",
        "w_indirect_1",
        "w_hyperdirect_0",
        "w_hyperdirect_1",
        "w_shortcut_0",
        "w_shortcut_1",
        "w_dopa_predict",
    ]

    for weight_var in weight_vars:
        aov = pg.rm_anova(
            dv=weight_var,
            within=["dbs_type", "dbs_state"],
            subject="sim_id",
            data=df_filtered,
            detailed=True,
        )
        print(f"ANOVA results for {weight_var}:\n", aov)
        print("\n")


def support_over_time(shortcut=True, for_selected=True):
    """
    Plots the support of the Thal neurons throughout the task for the different dbs types and shortcut modes.
    """
    # Load support data
    loaded_vars = load_variables(
        name_list=[
            f"support_values_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}"
            for shortcut in [0, 1]
            for dbs_state in range(6)
            for column in range(100)
        ],
        path="data/simulation_data/",
    )

    dbs_state_names = [
        "dbs-off",
        "suppression",
        "efferent",
        "afferent",
        "passing fibres GPe-STN",
        "dbs-comb",
    ]

    support_exc_plot_dict = {
        "dbs_state": [],
        "subject": [],
        "bin": [],
        "support": [],
    }
    support_inh_plot_dict = {
        "dbs_state": [],
        "subject": [],
        "bin": [],
        "support": [],
    }
    support_diff_plot_dict = {
        "dbs_state": [],
        "subject": [],
        "bin": [],
        "support": [],
    }
    support_indices = np.arange(120)
    for dbs_state in range(6):

        # skip dbs states affernet and passing fibres GPe-STN
        if dbs_state in [3, 4]:
            continue

        selection_data_df = load_data_previously_selected(
            subject_type="simulation",
            shortcut_type="plastic" if shortcut else "fixed",
            dbs_state="OFF" if dbs_state == 0 else "ON",
            dbs_variant=dbs_state_names[dbs_state],
            switch_choice_manipulation=None,
        )

        support_exc_all = np.zeros((100, 120))
        support_inh_all = np.zeros((100, 120))
        for subject in range(100):
            # extract the choices from the current subject from the selection_data_df
            # with columns subject, trial, choice,a nd reward
            choices = (
                selection_data_df[selection_data_df["subject"] == subject][
                    "choice"
                ].values
                - 1
            )
            # get the supports for action 0 for the current subject
            support_exc_for_0 = loaded_vars[
                f"support_values_Shortcut{int(shortcut)}_DBS_State{dbs_state}_sim{subject}"
            ]["support_exc"]
            support_inh_for_0 = loaded_vars[
                f"support_values_Shortcut{int(shortcut)}_DBS_State{dbs_state}_sim{subject}"
            ]["support_inh"]
            # get the supports for the selected action
            support_exc_for_selected = support_exc_for_0.copy()
            support_exc_for_selected[choices == 1] *= -1
            support_inh_for_selected = support_inh_for_0.copy()
            support_inh_for_selected[choices == 1] *= -1

            # store the support arrays and indice arrays centered around the change points
            support_exc_all[subject] = (
                support_exc_for_selected if for_selected else support_exc_for_0
            )
            support_inh_all[subject] = (
                support_inh_for_selected if for_selected else support_inh_for_0
            )

        # bin the trials and average over the values within each bin
        bin_size = 40 if for_selected else 5
        support_exc_binned = np.nanmean(
            support_exc_all.reshape(100, -1, bin_size),
            axis=2,
        )
        support_inh_binned = np.nanmean(
            support_inh_all.reshape(100, -1, bin_size),
            axis=2,
        )
        support_diff_binned = support_exc_binned - support_inh_binned
        # only take each 5th value of support_for_0_mean_indices
        support_indices_binned = support_indices[::bin_size]

        # for support_exc_plot_dict convert the support_exc with shape (n_subjects, n_bins)
        # to an dictionary in long format with keys "dbs_state", "subject", "bin", "support"
        subjects_long = np.repeat(np.arange(100), len(support_indices_binned))
        dbs_state_long = np.full(
            100 * len(support_indices_binned), dbs_state_names[dbs_state]
        )
        bin_long = np.tile(support_indices_binned, 100)
        support_exc_plot_dict["dbs_state"].extend(dbs_state_long)
        support_exc_plot_dict["subject"].extend(subjects_long)
        support_exc_plot_dict["bin"].extend(bin_long)
        support_exc_plot_dict["support"].extend(support_exc_binned.flatten())
        # same for support_inh_plot_dict
        support_inh_plot_dict["dbs_state"].extend(dbs_state_long)
        support_inh_plot_dict["subject"].extend(subjects_long)
        support_inh_plot_dict["bin"].extend(bin_long)
        support_inh_plot_dict["support"].extend(support_inh_binned.flatten())
        # same for support_diff_plot_dict
        support_diff_plot_dict["dbs_state"].extend(dbs_state_long)
        support_diff_plot_dict["subject"].extend(subjects_long)
        support_diff_plot_dict["bin"].extend(bin_long)
        support_diff_plot_dict["support"].extend(support_diff_binned.flatten())

    # convert the dictionaries to pandas dataframes
    support_exc_df = pd.DataFrame.from_dict(support_exc_plot_dict)
    support_inh_df = pd.DataFrame.from_dict(support_inh_plot_dict)

    # do plotting
    if for_selected:
        # use the data from support_exc_plot_dict to plot boxplots using seaborn with
        # x="bin", y="support", hue="dbs_state"
        plt.figure(figsize=(10, 15))
        # create three subplots for exc, inh support
        ax_exc = plt.subplot(211)
        ax_inh = plt.subplot(212)
        sns.boxplot(
            x="bin",
            y="support",
            hue="dbs_state",
            palette={
                "suppression": (1, 0.7, 0.7, 0.8),
                "efferent": (1, 0.5, 0.5, 0.8),
                "dbs-comb": (0.8, 0, 0, 0.8),
                "dbs-off": "darkblue",
            },
            data=support_exc_df,
            ax=ax_exc,
            showmeans=True,
            meanprops={
                "markerfacecolor": "black",
                "markeredgecolor": "white",
            },
            linecolor="black",
        )
        ax_exc.axhline(0, color="black", linestyle="--", alpha=0.5)
        # same for support_inh_plot_dict
        sns.boxplot(
            x="bin",
            y="support",
            hue="dbs_state",
            palette={
                "suppression": (1, 0.7, 0.7, 0.8),
                "efferent": (1, 0.5, 0.5, 0.8),
                "dbs-comb": (0.8, 0, 0, 0.8),
                "dbs-off": "darkblue",
            },
            data=support_inh_df,
            ax=ax_inh,
            showmeans=True,
            meanprops={
                "markerfacecolor": "black",
                "markeredgecolor": "white",
            },
            linecolor="black",
            legend=False,
        )
        ax_inh.axhline(0, color="black", linestyle="--", alpha=0.5)

        ax_exc.set_ylabel("shortcuts' support for selected")
        ax_inh.set_ylabel("basal ganglias' support for selected")
        # ax_exc legend position top right
        ax_exc.legend(loc="upper right")
        # equalize the axes ranges by first getting the current limits and then use the
        # maximum of the absolute values
        ax_exc_ylim = ax_exc.get_ylim()
        ax_inh_ylim = ax_inh.get_ylim()
        ax_exc.set_ylim(
            min(ax_exc_ylim[0], ax_inh_ylim[0]),
            max(ax_exc_ylim[1], ax_inh_ylim[1]),
        )
        ax_inh.set_ylim(
            min(ax_exc_ylim[0], ax_inh_ylim[0]),
            max(ax_exc_ylim[1], ax_inh_ylim[1]),
        )

        plt.tight_layout()
        plt.savefig(
            f"fig/__fig_support_for_selected_over_time_shortcut_{int(shortcut)}__.png",
            dpi=300,
        )
        plt.close("all")
    else:
        # combine the dataframes to one
        support_exc_df_dbs = support_exc_df.copy()
        support_exc_df_dbs["support_type"] = "shortcut"
        support_inh_df_dbs = support_inh_df.copy()
        support_inh_df_dbs["support_type"] = "basal ganglia"
        support_df_dbs = pd.concat([support_exc_df_dbs, support_inh_df_dbs])

        # use seaborns relplot to plot the data with x="bin", y="support",
        # hue="support_type", col="dbs_state"
        plt.figure(figsize=(15, 10))
        sns.relplot(
            x="bin",
            y="support",
            hue="support_type",
            col="dbs_state",
            kind="line",
            data=support_df_dbs,
            palette={"shortcut": "red", "basal ganglia": "blue"},
        )
        plt.savefig(
            f"fig/__fig_support_for_0_over_time_shortcut_{int(shortcut)}__.png", dpi=300
        )
        plt.close("all")
        return

    # run linear mixed effect model for each bin of support_exc_df and support_inh_df
    # fixed effect for dbs_state compared to baseline (dbs-off)
    for support_df_id, support_df in enumerate([support_exc_df, support_inh_df]):
        for bin in support_df["bin"].unique():
            data_df = support_df.copy()
            data_df = data_df[data_df["bin"] == bin]
            data_df["dbs_state"] = data_df["dbs_state"].astype("category")
            data_df["bin"] = data_df["bin"].astype("category")
            model = smf.mixedlm(
                "support ~ C(dbs_state, Treatment('dbs-off'))",
                data_df,
                groups=data_df["subject"],
            )
            result = model.fit()

            # save results
            with open(
                f"statistic/support_{['exc', 'inh'][support_df_id]}_difference_dbs_on_off_shortcut_{int(shortcut)}_bin_{bin}.txt",
                "w",
            ) as fh:
                fh.write(result.summary().as_text())


#################################################################################################################
########################################### function call #######################################################
#################################################################################################################


if __name__ == "__main__":
    if __fig_shortcut_on_off_line__:
        shortcut_on_off_line(14)

    if __fig_shortcut_on_off__:
        shortcut_on_off(True, 14)

    if __fig_dbs_on_off_14_and_100__:
        dbs_on_off_14_and_100(shortcut=True)
        dbs_on_off_14_and_100(shortcut=False)

    if __fig_activity_changes_dbs_on__:
        activity_changes_dbs_on()

    if __fig_activity_changes_dbs_off__:
        activity_changes_dbs_off()

    if __fig_gpi_scatter__:
        gpi_scatter()

    if __fig_load_simulate__:
        load_simulate()

    if __fig_dbs_parameter__:
        dbs_parameter()

    if __fig_load_simulate_dbscomb__:
        load_simulate_dbscomb()

    if __fig_parameter_gpi_inhib__:
        parameter_gpi_inhib()

    if __fig_weights_over_time__:
        weights_over_time()

    if __fig_support_over_time__:
        # analyze support for selected action for plastic and fixed shortcut
        support_over_time(shortcut=True)
        support_over_time(shortcut=False)
        # plot support for "action 0" for plastic shortcut
        support_over_time(shortcut=True, for_selected=False)
