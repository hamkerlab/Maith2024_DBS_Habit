import subprocess
import visualization as vis
import statistic as stat
from CompNeuroPy import run_script_parallel
import sys
import pandas as pd

N_JOBS = int(sys.argv[1])

###################################################################################################################
############################################### record data #######################################################
###################################################################################################################
# get simulation data
get_simulation_data = True

# get activity change data -> firing rates for one trial
get_activity_change_data = False

# get parameter data -> for suppression, efferent, afferent and passing-fibres
get_dbs_parameter_data = False

# get load simulation data
get_load_simulate_data = False

###################################################################################################################
############################################### visualization #####################################################
###################################################################################################################
# plot_figures = True -> create the images from the existing data without starting the simulation
plot_figures = False

fig_shortcut_on_off = True
fig_dbs_on_off_14_100 = True
fig_activity_changes_dbs_on = True
fig_activity_changes_dbs_off = True
fig_gpi_scatter = True
fig_load_simulate = True
fig_dbs_parameter = True
fig_parameter_gpi_inhib = True


###################################################################################################################
################################################### statistic #####################################################
###################################################################################################################
# run_statistic = True -> create statistic data without starting the simulation
run_statistic = False

check_H1 = True
check_H2 = True
check_H3 = True
anova_load_simulation = True
pairwise_ttest_load_simulation = True
previously_selected = True

# Save means and standard errors
save_mean_and_errors = True


#####################################################################################################
########################################### main function ###########################################
#####################################################################################################


def run_sim(parameter, step, dbs_param_state):
    """
    arg1 - number of people
    arg2 - DBS state
    arg3 - shortcut plastic/fixed
    arg4 - visualization of firing rates ON/OFF (old)
    arg5 - visualization of synapse weights ON/OFF (old)
    arg6 - save simulation data
    arg7 - dbs parameter
    arg8 - save dbs parameters
    arg9 - dbs parameter active step
    arg10 - save average rate GPi
    arg11 - dbs parameter state
    """

    if get_simulation_data:  # simulation data for 100 simulations
        dbs_states = 6
        shortcut = 2
        number_of_persons = 100
        dbs_param_state = 0
        dbs_alone = False
        shortcut_alone = False
        save_mean_GPi = False
        save_parameter_data = False
    else:  # parameter data for dbs_states (14 simulations)
        dbs_states = dbs_param_state
        shortcut = 2
        number_of_persons = 14
        dbs_alone = True
        shortcut_alone = True
        save_mean_GPi = True
        save_parameter_data = True

    vis_firing_rates = False
    vis_weigths = False
    skript_name = "simulation.py"

    if dbs_alone == True and shortcut_alone == True:
        # create args_list
        args_list = []
        for k in range(number_of_persons):
            arg1 = str(k)
            arg2 = str(dbs_states - 1)
            arg3 = str(shortcut - 1)
            arg4 = str(vis_firing_rates)
            arg5 = str(vis_weigths)
            arg6 = str(get_simulation_data)
            arg7 = str(parameter)
            arg8 = str(save_parameter_data)
            arg9 = str(step)
            arg10 = str(save_mean_GPi)
            arg11 = str(dbs_param_state)
            args = [
                arg1,
                arg2,
                arg3,
                arg4,
                arg5,
                arg6,
                arg7,
                arg8,
                arg9,
                arg10,
                arg11,
            ]
            args_list.append(args)
        run_script_parallel(script_path=skript_name, n_jobs=N_JOBS, args_list=args_list)
    else:
        # create args_list
        args_list = []
        for i in range(shortcut):
            for j in range(dbs_states):
                for k in range(number_of_persons):
                    arg1 = str(k)
                    arg2 = str(j)
                    arg3 = str(i)
                    arg4 = str(vis_firing_rates)
                    arg5 = str(vis_weigths)
                    arg6 = str(get_simulation_data)
                    arg7 = str(parameter)
                    arg8 = str(save_parameter_data)
                    arg9 = str(step)
                    arg10 = str(save_mean_GPi)
                    arg11 = str(dbs_param_state)
                    args = [
                        arg1,
                        arg2,
                        arg3,
                        arg4,
                        arg5,
                        arg6,
                        arg7,
                        arg8,
                        arg9,
                        arg10,
                        arg11,
                    ]
                    args_list.append(args)
        run_script_parallel(script_path=skript_name, n_jobs=N_JOBS, args_list=args_list)

    # combine the saved data which was previously created sequentially but now for
    # run_script_parallel was adjusted (save everything separately) and now needs to
    # be combined after all simulations

    simulation_data_combined = {}
    parameter_data_combined = {}
    # loop over all conducted simulations
    for args in args_list:
        (
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
        ) = args

        # simulation_data
        # check if save_data saved something (based on
        # if save_data == "True":
        # in simulation.py)
        if arg6 == "True":
            # load the file (name based on simulation.py) and add it to combined data
            # as if the combined data would be created sequentially
            shortcut = int(arg3)
            dbs_state = int(arg2)
            column = int(arg1)
            filepath = f"data/simulation_data/Results_Shortcut{shortcut}_DBS_State{dbs_state}_sim{column}.json"
            # the key is the file which was created sequentially before
            key = f"data/simulation_data/Results_Shortcut{shortcut}_DBS_State{dbs_state}.json"
            data = pd.read_json(filepath, orient="records", lines=True)
            if key not in simulation_data_combined.keys():
                simulation_data_combined[key] = pd.DataFrame({})
            simulation_data_combined[key][column] = data[0]

        # parameter_data (save_parameter fucntion from simulation.py)
        # check if save_parameter saved something (based on
        # if save_parameter_data == "True" and dbs_state > 0 and dbs_state < 5:
        # in simulation.py)
        dbs_state = int(arg2)
        column = int(arg1)
        step = int(arg9)
        save_parameter_data = arg8
        if save_parameter_data == "True" and dbs_state > 0 and dbs_state < 5:
            # Results files (see save_parameter function from simulation.py)
            if dbs_state == 1:
                filepath = f"data/parameter_data/1_suppression/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
                key = f"data/parameter_data/1_suppression/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 2:
                filepath = f"data/parameter_data/2_efferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
                key = f"data/parameter_data/2_efferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 3:
                filepath = f"data/parameter_data/3_afferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
                key = f"data/parameter_data/3_afferent/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}.json"
            if dbs_state == 4:
                filepath = f"data/parameter_data/4_passing_fibres/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}_sim{column}.json"
                key = f"data/parameter_data/4_passing_fibres/Results_Shortcut{shortcut}_DBS_State{dbs_state}_Step{step}.json"

            data = pd.read_json(filepath, orient="records", lines=True)
            if key not in parameter_data_combined.keys():
                parameter_data_combined[key] = pd.DataFrame({})
            parameter_data_combined[key][column] = data[0]

            # Param files (see save_parameter function from simulation.py)
            if column == 0:
                if dbs_state == 1:
                    filepath = f"data/parameter_data/1_suppression/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
                    key = f"data/parameter_data/1_suppression/Param_Shortcut{shortcut}_DBS_State{dbs_state}.json"
                if dbs_state == 2:
                    filepath = f"data/parameter_data/2_efferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
                    key = f"data/parameter_data/2_efferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}.json"
                if dbs_state == 3:
                    filepath = f"data/parameter_data/3_afferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
                    key = f"data/parameter_data/3_afferent/Param_Shortcut{shortcut}_DBS_State{dbs_state}.json"
                if dbs_state == 4:
                    filepath = f"data/parameter_data/4_passing_fibres/Param_Shortcut{shortcut}_DBS_State{dbs_state}_step{step}.json"
                    key = f"data/parameter_data/4_passing_fibres/Param_Shortcut{shortcut}_DBS_State{dbs_state}.json"

                data = pd.read_json(filepath, orient="records", lines=True)
                if key not in parameter_data_combined.keys():
                    parameter_data_combined[key] = pd.DataFrame({})
                parameter_data_combined[key][step] = data[0]

    # save simulation data combined
    for key, val in simulation_data_combined.items():
        val.to_json(
            key,
            orient="records",
            lines=True,
        )

    # save parameter data combined
    for key, val in parameter_data_combined.items():
        val.to_json(
            key,
            orient="records",
            lines=True,
        )

    # TODO  fix the data save_mean_GPi


#####################################################################################################
########################################## parameter data ###########################################
#####################################################################################################


def run_parameter():

    ########################## initial parameter ################################################

    for i in range(4):
        # parameter settings for dbs_states

        if i == 0:  # suppression
            min = 0
            max = 0.7
            intervall = 0.01
            print("save parameter data suppression...", "\n")
        elif i == 1:  # efferent
            min = 0
            max = 0.7
            intervall = 0.01
            print("save parameter data efferent...", "\n")
        elif i == 2:  # afferent
            min = 0
            max = 0.3
            intervall = 0.005
            print("save parameter data afferent...", "\n")
        elif i == 3:  # passing-fibres
            min = 0
            max = 30
            intervall = 0.5
            print("save parameter data passing-fibres...", "\n")

        # init parameter
        dbs_param_state = i + 2
        parameter = min
        step = int((max - min) / intervall) + 2

        for j in range(step):
            ############################### change parameter #########################################
            if j > 0:
                parameter = parameter + intervall

            print("parameter:", parameter)

            run_sim(parameter, j, dbs_param_state)


#####################################################################################################
###################################### get activity changes data ####################################
#####################################################################################################


def run_activity_change():
    dbs = 6
    shortcut = 2
    number_of_persons = 100
    sessions = 2
    boxplots = False

    for i in range(sessions):
        for j in range(dbs):
            for n in range(number_of_persons):
                arg1 = str(j)
                arg2 = str(shortcut - 1)
                arg3 = str(i)
                arg4 = str(boxplots)
                arg5 = str(n)
                command = [
                    "python",
                    "simulation_activity_change.py",
                    arg1,
                    arg2,
                    arg3,
                    arg4,
                    arg5,
                ]
                subprocess.run(command)


#####################################################################################################
#################################### load simulation data ###########################################
#####################################################################################################


def run_load_simulation():
    """
    conditions:
    1 - tests saved data without learning
    2 - simulate on / load on
    3 - simulate on / load off
    4 - simulate off / load on
    5 - simulate off / load off
    """

    vis_firing_rates = False
    vis_weigths = False

    dbs_state = 6
    short = 2
    number_of_persons = 100

    for i in range(2):
        if i == 0:
            save_load_simulate_data = True
            conditions = 1
        else:
            save_load_simulate_data = False
            conditions = 5

        for condition in range(conditions):
            for j in range(dbs_state):
                for k in range(number_of_persons):

                    if save_load_simulate_data == False and conditions == 1:
                        continue

                    arg1 = str(k)
                    arg2 = str(j)
                    arg3 = str(short - 1)
                    arg4 = str(vis_firing_rates)
                    arg5 = str(vis_weigths)
                    arg6 = str(save_load_simulate_data)
                    arg7 = str(condition + 1)
                    command = [
                        "python",
                        "load_simulation.py",
                        arg1,
                        arg2,
                        arg3,
                        arg4,
                        arg5,
                        arg6,
                        arg7,
                    ]
                    subprocess.run(command)


#####################################################################################################
######################################## function call ##############################################
#####################################################################################################

if (
    get_dbs_parameter_data == False
    and get_simulation_data == True
    and get_activity_change_data == False
    and plot_figures == False
    and run_statistic == False
    and get_load_simulate_data == False
):
    parameter = str(0)
    run_sim(parameter, 0, 0)
if (
    get_dbs_parameter_data == False
    and get_activity_change_data == True
    and get_simulation_data == False
    and plot_figures == False
    and run_statistic == False
    and get_load_simulate_data == False
):
    run_activity_change()
if (
    get_dbs_parameter_data == True
    and get_activity_change_data == False
    and get_simulation_data == False
    and plot_figures == False
    and run_statistic == False
    and get_load_simulate_data == False
):
    run_parameter()
if (
    get_dbs_parameter_data == False
    and get_activity_change_data == False
    and get_simulation_data == False
    and plot_figures == False
    and run_statistic == False
    and get_load_simulate_data == True
):
    run_load_simulation()

#####################################################################################################
################################# function call - visualization #####################################
#####################################################################################################

if plot_figures:
    if fig_shortcut_on_off:
        vis.shortcut_on_off(True, 14)

    if fig_dbs_on_off_14_100:
        vis.dbs_on_off_14_and_100(True)

    if fig_activity_changes_dbs_on:
        vis.activity_changes_dbs_on()

    if fig_activity_changes_dbs_off:
        vis.activity_changes_dbs_off()

    if fig_gpi_scatter:
        vis.gpi_scatter()

    if fig_load_simulate:
        vis.load_simulate()

    if fig_dbs_parameter:
        vis.dbs_parameter()

    if fig_parameter_gpi_inhib:
        vis.parameter_gpi_inhib()


#####################################################################################################
#################################### function call - statistic ######################################
#####################################################################################################

if run_statistic:
    if check_H1 or check_H2 or check_H3:
        stat.run_statistic(check_H1, check_H2, check_H3, 14)
        stat.run_statistic(check_H1, check_H2, check_H3, 100)

    if anova_load_simulation:
        stat.anova_load_simulation(100)

    if pairwise_ttest_load_simulation:
        stat.anova_load_simulation_ttest(100)

    if save_mean_and_errors:
        stat.save_mean_error(14)
        stat.save_mean_error(100)

    if previously_selected:
        stat.previously_selected()
