import matplotlib.pyplot as plt
import numpy as np
import statistic as stat
from CompNeuroPy import load_variables

#################################################################################################################
########################################### plot figures ########################################################
#################################################################################################################

__fig_shortcut_on_off__ = False
__fig_dbs_on_off_14_and_100__ = False
__fig_activity_changes_dbs_on__ = False
__fig_activity_changes_dbs_off__ = False
__fig_gpi_scatter__ = False
__fig_load_simulate__ = False
__fig_dbs_parameter__ = False
__fig_parameter_gpi_inhib__ = False
__fig_weights_over_time__ = True


##############################################################################
############################# scale y-axis ###################################
##############################################################################

min_y_reward = 0
max_y_reward = 30

min_y_habit = 5
max_y_habit = 30

label_size = 9

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

    ############################################### histo settings ##############################################

    # sessions
    session = ["1", "2", "3"]
    x = np.arange(len(session)) * 1.5

    # bar width
    width = 0.3

    # bar colors
    colors = ["darkblue", "steelblue", "lightblue"]

    # bar positions
    positions = [x - 1.3 * width, x, x + 1.3 * width]

    labels = ["DBS OFF Patients", "Model Plastic Shortcut", "Model Fixed Shortcut"]

    fig, ax = plt.subplots(figsize=(3.4, 3.4))

    ################################################## plot bars #################################################

    for i in range(len(means)):
        # plot means
        ax.bar(
            positions[i],
            means[i],
            width=width,
            color=colors[i],
        )

    # plot legend
    ax.legend(labels, fontsize="small")

    for i in range(len(means)):
        # plot error bars
        ax.errorbar(
            positions[i],
            means[i],
            yerr=standarderrors[i],
            fmt="none",
            color="black",
            capsize=2,
        )

    ################################################### axis labels ###############################################

    # x-axis
    plt.xticks(x, session)

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

    plt.tight_layout()

    # save fig
    plt.savefig("fig/__fig_shortcut_on_off__.png", dpi=300)
    plt.savefig("fig/__fig_shortcut_on_off__.svg", format="svg", dpi=300)

    plt.show()


#################################################################################################################
################################## __fig_dbs_on_off_14_and_100__ ################################################
#################################################################################################################


def dbs_on_off_14_and_100(switch):

    ################################################# load data #################################################
    # load simulation data
    filepath1 = "data/simulation_data/Results_Shortcut1_DBS_State1.json"
    filepath2 = "data/simulation_data/Results_Shortcut1_DBS_State2.json"
    # filepath3 = "data/simulation_data/Results_Shortcut1_DBS_State3.json"
    filepath4 = "data/simulation_data/Results_Shortcut1_DBS_State4.json"
    filepath5 = "data/simulation_data/Results_Shortcut1_DBS_State5.json"
    filepath6 = "data/simulation_data/Results_Shortcut1_DBS_State0.json"

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

    ############################################### histo settings ##############################################

    # sessions
    session = ["1", "2", "3"]
    x = np.arange(len(session)) * 2

    # bar width
    width = 0.2

    # bar colors
    patient_colors = [(0.8, 0, 0, 0.7), "darkblue"]
    simulation_colors = [
        (1, 0.7, 0.7, 0.8),  # very bright red
        (1, 0.5, 0.5, 0.8),  # light red
        (1, 0.3, 0.3, 0.8),  # red
        (0.8, 0, 0, 0.8),  # dark red
        "darkblue",
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
        "dbs-all",
        # "dbs-off",
    ]

    # plot size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))

    ################################################## plot bars #################################################

    ############ patient data ################
    for i in range(len(means)):
        # plot means
        ax1.bar(
            patient_positions[i],
            means[i],
            width=width,
            color=patient_colors[i],
        )

    # plot legend
    ax1.legend(patient_labels, fontsize="x-small", loc="upper left")

    for i in range(len(means)):
        # plot error bars
        ax1.errorbar(
            patient_positions[i],
            means[i],
            yerr=standarderror[i],
            fmt="none",
            color="black",
            capsize=2,
        )

    # ax1.set_xlabel("session")
    ax1.set_ylim(5, 25)  # Skalierung der y-Achse
    if switch:
        ax1.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax1.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ############ simulation data N=14 ################
    for i in range(len(means_14)):
        # plot means
        ax2.bar(
            simulation_positions[i],
            means_14[i],
            width=width,
            color=simulation_colors[i],
        )

    # plot legend
    ax2.legend(simulation_labels, fontsize="x-small", loc="upper left")

    for i in range(len(means_14)):
        # plot error bars
        ax2.errorbar(
            simulation_positions[i],
            means_14[i],
            yerr=standarderror_14[i],
            fmt="none",
            color="black",
            capsize=2,
        )

    # ax2.set_xlabel("session")
    ax2.set_ylim(5, 25)  # Skalierung der y-Achse
    if switch:
        ax2.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax2.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ############ simulation data N=100 ################
    for i in range(len(means_100)):
        # plot means
        ax3.bar(
            simulation_positions[i],
            means_100[i],
            width=width,
            color=simulation_colors[i],
        )

    # plot legend
    # ax3.legend(simulation_labels, fontsize="large", loc="upper left")

    for i in range(len(means_100)):
        # plot error bars
        ax3.errorbar(
            simulation_positions[i],
            means_100[i],
            yerr=standarderror_100[i],
            fmt="none",
            color="black",
            capsize=2,
        )

    # settings axis
    ax3.set_xlabel("Session", fontweight="bold", fontsize=label_size)
    ax3.set_ylim(5, 25)  # Skalierung der y-Achse
    if switch:
        ax3.set_ylabel("unrewarded\ndecisions", fontweight="bold", fontsize=label_size)
    else:
        ax3.set_ylabel("rewarded\ndecisions", fontweight="bold", fontsize=label_size)

    ################################################### axis labels ###############################################

    # Gemeinsame x-Achsen-Ticks für alle Subplots
    plt.setp([ax1, ax2, ax3], xticks=x, xticklabels=session)

    # same x-axis in all plots
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.set_xticks(x)
    ax3.set_xticklabels(session)

    ax1.set_xlim(ax3.get_xlim())
    ax2.set_xlim(ax3.get_xlim())
    ax3.set_xlim(ax1.get_xlim())

    plt.tight_layout

    # plot labels
    plt.text(0.01, 0.89, "A", transform=plt.gcf().transFigure, fontsize=11)
    plt.text(0.01, 0.62, "B", transform=plt.gcf().transFigure, fontsize=11)
    plt.text(0.01, 0.35, "C", transform=plt.gcf().transFigure, fontsize=11)

    # save fig
    plt.savefig("fig/__fig_dbs_on_off_14_and_100__.png", dpi=300)
    plt.savefig("fig/__fig_dbs_on_off_14_and_100__.svg", format="svg", dpi=300)

    plt.show()


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

        # print("\n", data_dbs1, "\n")

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

        # mean_dbs2 = np.array(mean_dbs2 / mean_dbs1

        # print("\n", mean_dbs1, "\n")

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

        label_y = [
            "$\\mathbf{IT}$",
            "$\\mathbf{StrD1}$",
            "$\\mathbf{StrD2}$",
            "$\\mathbf{STN}$",
            "$\\mathbf{GPi}$",
            "$\\mathbf{GPe}$",
            "$\\mathbf{Thalamus}$",
            "$\\mathbf{PFC}$",
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
                    yerr=error_dbs2[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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
                    yerr=error_dbs3[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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

        # title
        axs[1].set_title("efferent", fontweight="bold", fontsize=label_size)

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
                    yerr=error_dbs4[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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

        # title
        axs[2].set_title("afferent", fontweight="bold", fontsize=label_size)

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
                    yerr=error_dbs5[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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

        # title
        axs[3].set_title("passing-fibres", fontweight="bold", fontsize=label_size)

        ####################### bds-all data ################################
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
                    yerr=error_dbs6[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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

        # title
        axs[4].set_title("dbs-all", fontweight="bold", fontsize=label_size)

        # Adjust layout to minimize margins
        # plt.subplots_adjust(left=0.07, right=0.96, top=0.9, bottom=0.15, wspace=0.2)

        plt.tight_layout()

        # save fig
        if session == 0:
            plt.savefig("fig/__fig_activity_change_dbs_on_init__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_on_init__.svg", format="svg", dpi=300
            )
        else:
            plt.savefig("fig/__fig_activity_change_dbs_on_learn__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_on_learn__.svg", format="svg", dpi=300
            )

        plt.show()


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

        colors = ["darkblue"]  # [(0.2, 0.2, 1.0, 0.7)]

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
            "$\\mathbf{IT}$",
            "$\\mathbf{StrD1}$",
            "$\\mathbf{StrD2}$",
            "$\\mathbf{STN}$",
            "$\\mathbf{GPi}$",
            "$\\mathbf{GPe}$",
            "$\\mathbf{Thalamus}$",
            "$\\mathbf{PFC}$",
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
                    yerr=error_dbs1[i],
                    fmt="none",
                    color="black",
                    # capsize=2,
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

        plt.tight_layout()

        # save fig
        if session == 0:
            plt.savefig("fig/__fig_activity_change_dbs_off_init__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_off_init__.svg", format="svg", dpi=300
            )
        else:
            plt.savefig("fig/__fig_activity_change_dbs_off_learn__.png", dpi=300)
            plt.savefig(
                "fig/__fig_activity_change_dbs_off_learn__.svg", format="svg", dpi=300
            )

        plt.show()


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
        fontsize="small",
    )

    plt.ylim(14, 35)
    plt.xlim(-0.05, 0.6)
    plt.tight_layout()

    plt.savefig("fig/__fig_gpi_scatter__.png", dpi=300)
    plt.savefig("fig/__fig_gpi_scatter__.svg", format="svg", dpi=300)
    plt.show()


#################################################################################################################
########################################## __fig_load_simulate__ ################################################
#################################################################################################################


def load_simulate():

    # number of simulations
    number_of_simulations = 100
    passingoff = True

    ################################ load dbs-on data ###########################################
    resultON = []

    for i in range(1, 6):
        # without afferent and passing-fibres
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

    if passingoff:
        index = 3
    else:
        index = 4

    result = []
    for i in range(index):
        result.append([resultOFF[i], resultSim[i], resultLoad[i], resultON[i]])

    ################################## boxplot settings #########################################

    if passingoff:
        # Einstellung
        colors = [
            (1, 0.9, 0.9, 0.7),  # very bright red
            (1, 0.6, 0.6, 0.7),  # bright red
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
        legend = ["suppression", "efferent", "dbs-all"]
    else:
        legend = ["suppression", "efferent", "passing-fibres", "dbs-all"]

    # plot size
    fig, ax = plt.subplots(figsize=(5, 3))

    # sessions
    session = ["off\noff", "on\noff", "off\non", "on\non"]

    # bar positions
    x = np.arange(len(session))

    if passingoff:
        distance = 1.4
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
        -4.5,
        "simulate\nload       ",
        fontsize=10,
        ha="center",
        va="center",
    )

    # legend
    legend_bars = [
        plt.bar(0, 0, color=colors[i], label=legend[i]) for i in range(len(colors))
    ]

    plt.legend(handles=legend_bars, loc="upper left", fontsize="small")

    # setting y-axis
    plt.ylabel("unrewarded decisions", fontweight="bold", fontsize=label_size)
    plt.ylim(0, 45)

    plt.tight_layout()

    plt.savefig("fig/__fig_load_simulate__.png", dpi=300)
    plt.savefig("fig/__fig_load_simulate__.svg", format="svg", dpi=300)

    plt.show()


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
                label=f"Session {j+1}",
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
        if i == "efferent":
            ax.legend(fontsize="xx-small")

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

        idx += 1
        plt.ylim(0.5, 1.4)

    plt.tight_layout()
    plt.savefig("fig/__fig_dbs_parameter__.png", dpi=300)
    plt.savefig("fig/__fig_dbs_parameter__.svg", format="svg", dpi=300)
    plt.show()


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
        ax2.set_xlabel(parameter_name[idx - 1])
        ax2.set_title(i, fontweight="bold", fontsize=label_size)
        if i == "efferent" or i == "passing fibres GPe-STN":
            ax2.set_ylabel("average rate", fontweight="bold", fontsize=label_size)
        else:
            ax2.set_yticklabels([])

        # legend
        lines = [line1, line2, line3]
        if i == "efferent":
            ax1.legend(lines, label, loc="upper right", fontsize="xx-small")

        # scale x-axis
        if idx == 1:
            ax1.set_xticks(np.arange(0, 0.70, 0.2))
        if idx == 2:
            ax2.set_xticks(np.arange(0, 0.70, 0.2))
        if idx == 3:
            ax1.set_xticks(np.arange(0, 0.31, 0.1))
        if idx == 4:
            ax2.set_xticks(np.arange(0, 1.60, 0.5))

        # min/max y-axis
        ax1.set_ylim(0.5, 1.4)
        ax2.set_ylim(-0.1, 0.7)

        idx += 1

    plt.tight_layout()
    plt.savefig("fig/__appendix_fig_parameter_gpi_inhib__.png", dpi=300)
    plt.savefig("fig/__appendix_fig_parameter_gpi_inhib__.svg", format="svg", dpi=300)
    plt.show()


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


def weights_over_time():
    # we have weights for each sim (0-99), shortcut (0-fixed,1-plastic), dbs state
    # (0-DBS-OFF,1-suppression,2-efferent,3-afferent,4-passing,5-dbs-all)

    # data/simulation_data/plastic_weights_Shortcut0_DBS_State0_sim0.pkl
    n_sims = 100
    n_dbs = 6
    n_trials = 120
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
    w_direct = np.mean(w_direct, axis=4, keepdims=True)  # IT is pre
    w_indirect = np.mean(w_indirect, axis=4, keepdims=True)  # IT is pre
    w_hyperdirect = np.mean(w_hyperdirect, axis=4, keepdims=True)  # IT is pre
    w_shortcut = np.mean(w_shortcut, axis=4, keepdims=True)  # IT is pre
    w_dopa_predict = np.mean(w_dopa_predict, axis=4, keepdims=True)  # StrD1 is pre

    # average over sim_ids the reversal order is the same for all sims
    # (6,120,2,1)
    w_direct = np.mean(w_direct, axis=0)
    w_indirect = np.mean(w_indirect, axis=0)
    w_hyperdirect = np.mean(w_hyperdirect, axis=0)
    w_shortcut = np.mean(w_shortcut, axis=0)
    # (6,120,1,1)
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
        ["dbs-all", True],
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

    plt.show()


#################################################################################################################
########################################### function call #######################################################
#################################################################################################################

if __name__ == "__main__":
    if __fig_shortcut_on_off__:
        shortcut_on_off(True, 14)

    if __fig_dbs_on_off_14_and_100__:
        dbs_on_off_14_and_100(True)

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

    if __fig_parameter_gpi_inhib__:
        parameter_gpi_inhib()

    if __fig_weights_over_time__:
        weights_over_time()
