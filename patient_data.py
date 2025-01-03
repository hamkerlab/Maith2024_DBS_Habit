import scipy.io as sio
import numpy as np
import pandas as pd


class Data:
    #############################################################################################################
    ############################################# initialization step ###########################################
    #############################################################################################################

    def __init__(
        self,
        data_folder="./Data_experimental_study/osfstorage-archive/Behavioural_data/",
        nr_subjects=19,
    ):
        self.data_folder = data_folder
        self.nr_subjects = nr_subjects
        self._data_dict = {}

    #############################################################################################################
    ############################### loading outcome data from 14 patients ########################################
    #############################################################################################################

    def load_outcome(self, sub_id=0, data_folder=None):

        for dbs in range(2):

            Result = []  # Initialize Result as a list

            if dbs == 0:
                dbs_mode = "ON"
            else:
                dbs_mode = "OFF"

            for sub_id in range(19):

                print(dbs_mode)

                # exclude persons with fault data
                # if sub_id == 4:
                # continue
                if sub_id == 7:
                    continue
                if sub_id == 10:
                    continue
                if sub_id == 11:
                    continue
                if sub_id == 13:
                    continue
                if sub_id == 15:
                    continue

                # set path for dataset
                file_name_results = f"P{sub_id}_RESULTS_DBS_{dbs_mode}.mat"

                # try to load data
                try:
                    data_sum_results = sio.loadmat(data_folder + file_name_results)
                except FileNotFoundError:
                    print(f"Could not load data from {file_name_results}")
                    return None

                # ectract outcome-data in array
                outcome = np.asarray(data_sum_results["outcome"]).squeeze()
                Result.append(outcome)

            # convert result in DataFrame
            df = pd.DataFrame(Result)

            # transpone DataFrame in format 19x120
            df = df.transpose()

            # save data as JSON-file
            filepath = f"data/patient_data/RewardsPerSession_{dbs_mode}_line.json"
            df.to_json(filepath, orient="records", lines=True)

    #############################################################################################################
    ############################### Load number of completed attempts per session ###############################
    #############################################################################################################

    def load_completed_tasks(
        self,
        data_folder=None,
        sub_id=0,
        dbs_mode="ON",
    ):

        data_folder = self.data_folder if data_folder is None else data_folder

        if f"{data_folder}_{sub_id}_{dbs_mode}" in self._data_dict:
            return self._data_dict[f"{data_folder}_{sub_id}_{dbs_mode}"]

        # filename
        file_name_results = f"P{sub_id}RT_DBS_{dbs_mode}.mat"

        # filepath
        try:
            data_results = sio.loadmat(data_folder + file_name_results)
        except FileNotFoundError:
            print(f"Could not load data from {file_name_results}")
            return None

        # return data as a dictionary
        self._data_dict[f"{data_folder}_{sub_id}_{dbs_mode}"] = data_results

        return data_results

    #############################################################################################################
    ######################## calculate the total number of completed attempts per session #######################
    #############################################################################################################

    def sum_completed_tasks(self, data_results, do_print=False):

        # array with data session1
        RT1 = np.array(data_results["RT1"])
        RT1 = np.isnan(RT1)
        RT1 = np.count_nonzero(~RT1)  # count completed tasks
        if RT1 < 20.0:
            RT1 = 0

        # array with data session2
        RT2 = np.array(data_results["RT2"])
        RT2 = np.isnan(RT2)
        RT2 = np.count_nonzero(~RT2)
        if RT2 < 20.0:
            RT2 = 0

        # array with data session3
        RT3 = np.array(data_results["RT3"])
        RT3 = np.isnan(RT3)
        RT3 = np.count_nonzero(~RT3)
        if RT3 < 20.0:
            RT3 = 0

        # return array with number of completed tasks per session
        return np.array(
            [
                RT1,
                RT2,
                RT3,
            ]
        )

    #############################################################################################################
    ##################### display and save the total number of completed tasks per session ###################
    #############################################################################################################

    def save_sum_completed_tasks(self):

        data_anz_tasks = np.full((self.nr_subjects, 2, 3), 0)

        # Loop through all 19 subjects and DBS-ON/OFF states
        for sub_id in range(self.nr_subjects - 1):
            for dbs_mode_idx, dbs_mode in enumerate(["ON", "OFF"]):

                # Load data indicating whether a trial was completed or not finished
                data_results = self.load_completed_tasks(
                    sub_id=sub_id, dbs_mode=dbs_mode
                )

                # If loading fails, proceed with the next subject/state
                if data_results is None:
                    continue

                # Function to sum up the completed trials per session
                data_anz_tasks[sub_id, dbs_mode_idx] = self.sum_completed_tasks(
                    data_results
                )
                data_anz_tasks[18, :] = (
                    40.0  # In study data, RT1, RT2, and RT3 are missing for Subject 18 -> manually extracted
                )

        DataON = data_anz_tasks[:, 0]
        DataOFF = data_anz_tasks[:, 1]

        ###################### Output array with the number of completed trials per session ###################

        print("DBS ON")
        print(DataON)
        print("")
        print("DBS OFF")
        print(DataOFF)
        print("")

        ########## Save array with the number of completed trials per session to an Excel file ###############

        ## Save data ON
        df = pd.DataFrame(data_anz_tasks[:, 0])
        filepath = "data/patient_data/Anz_CompleteTasks_ON.json"
        df.to_json(filepath, orient="records", lines=True)

        ## Save data OFF
        df = pd.DataFrame(data_anz_tasks[:, 1])
        filepath = "data/patient_data/Anz_CompleteTasks_OFF.json"
        df.to_json(filepath, orient="records", lines=True)

    #############################################################################################################
    ###################################### loading data from 19 patients ########################################
    #############################################################################################################

    def load_rewards(
        self,
        data_folder=None,
        sub_id=0,
        dbs_mode="ON",
    ):

        ### set data folder
        data_folder = self.data_folder if data_folder is None else data_folder

        ### set path to data
        file_name_results = f"P{sub_id}_RESULTS_DBS_{dbs_mode}.mat"

        ### try to load data
        try:
            data_sum_results = sio.loadmat(data_folder + file_name_results)
        except FileNotFoundError:
            ### if loading data fails, print warning and return None
            print(f"Could not load data from {file_name_results}")
            return None

        ### return data and store it in dictionary
        self._data_dict[f"{data_folder}_{sub_id}_{dbs_mode}"] = data_sum_results

        return data_sum_results

    #############################################################################################################
    ##################################### calculate total rewards per session ###################################
    #############################################################################################################

    def sum_rewards(self, data_sum_results, anzS1, anzS2, anzS3, do_print=False):

        # print("\n", data_sum_results, "\n")

        outcome_arr = np.array(data_sum_results["outcome"][0])

        if (
            len(outcome_arr[:anzS1]) > 20
        ):  # test whether half of attempts have been completed -> otherwise NaN
            reward_session1 = np.sum(outcome_arr[:anzS1])  # sum rewards
        else:
            reward_session1 = np.nan

        if len(outcome_arr[anzS1:anzS2]) > 20:
            reward_session2 = np.sum(outcome_arr[anzS1:anzS2])
        else:
            reward_session2 = np.nan

        if len(outcome_arr[anzS2:anzS3]) > 20:
            reward_session3 = np.sum(outcome_arr[anzS2:anzS3])
        else:
            reward_session3 = np.nan

        # reward number per session
        return np.array(
            [
                reward_session1,
                reward_session2,
                reward_session3,
            ]
        )

    #############################################################################################################
    ############################################# save and load data ############################################
    #############################################################################################################

    def save_sum_rewards(self):
        ####################################### read data #######################################################

        data_sum_rewards = np.full((self.nr_subjects, 2, 3), np.nan)

        for sub_id in range(self.nr_subjects):
            for dbs_mode_idx, dbs_mode in enumerate(["ON", "OFF"]):

                data_sum_results = self.load_rewards(sub_id=sub_id, dbs_mode=dbs_mode)

                # check data
                if data_sum_results is None:
                    continue

                ######################### completed tasks per session ############################################

                number = pd.read_json(
                    f"data/patient_data/Anz_CompleteTasks_{dbs_mode}.json",
                    orient="records",
                    lines=True,
                )
                number = number.to_numpy()

                ################################### rewards per session #################################

                number = number[sub_id, :]
                numberS1 = int(number[0])
                numberS2 = int(number[0]) + int(number[1])
                numberS3 = int(number[0]) + int(number[1]) + int(number[2])

                data_sum_rewards[sub_id, dbs_mode_idx] = self.sum_rewards(
                    data_sum_results, numberS1, numberS2, numberS3
                )

        DataON = data_sum_rewards[:, 0]
        DataOFF = data_sum_rewards[:, 1]

        ######################################## exclude people from analysis ###################################
        """
        In the study by de A Marcelino, one person was excluded because they had not completed more than
        around half of the tasks in the DBS-ON state.
        -> Person 10 has the fewest completed tasks with 72 out of 120 and is excluded
        -> this leaves 14 DBS-ON and 18 DBS-OFF subjects as in the study
        -> second person only for testing
        """

        # exclude first people
        PersonNr_delete1 = 10
        delete1_on_off = True
        if delete1_on_off == True:
            DataON = np.delete(DataON, PersonNr_delete1, axis=0)
            DataOFF = np.delete(DataOFF, PersonNr_delete1, axis=0)

        # exclude second people
        PersonNr_delete2 = 4
        delete2_an_aus = False
        if delete2_an_aus == True:
            DataON = np.delete(DataON, PersonNr_delete2, axis=0)
            DataOFF = np.delete(DataOFF, PersonNr_delete2, axis=0)

        ################################### print rewards per session ###################################

        ### print data
        print("DBS ON")
        print(DataON)
        print("")
        print("DBS OFF")
        print(DataOFF)
        print("")

        ############################################ save data ############################################

        # save DBS-ON data
        df = pd.DataFrame(DataON)
        filepath = "data/patient_data/RewardsPerSession_ON.json"
        df.to_json(filepath, orient="records", lines=True)

        # save DBS-OFF data
        df = pd.DataFrame(DataOFF)
        filepath = "data/patient_data/RewardsPerSession_OFF.json"
        df.to_json(filepath, orient="records", lines=True)

    def choices_rewards_per_trial(self):
        ####################################### read data #######################################################

        ### sessions are not always 120 trials long, so use list
        choices_rewards_dict = {}
        not_use_patients = [10]

        for sub_id in range(self.nr_subjects):
            choices_rewards_dict[sub_id] = {}
            for _, dbs_mode in enumerate(["ON", "OFF"]):

                # load data
                data_sum_results = self.load_rewards(sub_id=sub_id, dbs_mode=dbs_mode)

                # check data, only collect DBS ON data
                if data_sum_results is None:
                    not_use_patients.append(sub_id)
                    continue

                choices = data_sum_results["choice"][0]
                rewards = data_sum_results["outcome"][0]
                choices_rewards_dict[sub_id][dbs_mode] = {
                    "choices": choices,
                    "rewards": rewards,
                }

        ### remove all persons that should not be used
        for patient in not_use_patients:
            choices_rewards_dict.pop(patient)

        ### save data using pickle
        import pickle

        with open("data/patient_data/choices_rewards_per_trial.pkl", "wb") as f:
            pickle.dump(choices_rewards_dict, f)

    #############################################################################################################
    ############################################## main - function ##############################################
    #############################################################################################################


if __name__ == "__main__":
    data_folder = "./Data_experimental_study/osfstorage-archive/Behavioural_data/"
    nr_subjects = 19
    data = Data(data_folder=data_folder, nr_subjects=nr_subjects)

    data.load_outcome(sub_id=0, data_folder=data_folder)
    data.save_sum_completed_tasks()
    data.save_sum_rewards()
    data.choices_rewards_per_trial()
