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
    ############################### Anzahl abgeschlossener Versuche pro Sitzung laden ###########################
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

        # Dateiname der geladen werden soll
        file_name_results = f"P{sub_id}RT_DBS_{dbs_mode}.mat"

        # Dateipfad der geladen werden soll
        try:
            data_results = sio.loadmat(data_folder + file_name_results)
        except FileNotFoundError:
            print(f"Could not load data from {file_name_results}")
            return None

        # Daten zurückgeben als Dictionary
        self._data_dict[f"{data_folder}_{sub_id}_{dbs_mode}"] = data_results

        return data_results

    #############################################################################################################
    ############################# Summe abgeschlossener Versuche pro Sitzung berechnen ##########################
    #############################################################################################################

    def sum_completed_tasks(self, data_results, do_print=False):

        # Array mit Daten für Sitzung 1 erstellen (Wert = Versuch abgeschlossen , Nan = Versuch nicht beendet)
        RT1 = np.array(data_results["RT1"])
        RT1 = np.isnan(RT1)
        RT1 = np.count_nonzero(
            ~RT1
        )  # Anzahl der abgeschlossenen Versuche in Sitzung 1 zählen
        if RT1 < 20.0:
            RT1 = 0

        # Array mit Daten für Sitzung 2 erstellen
        RT2 = np.array(data_results["RT2"])
        RT2 = np.isnan(RT2)
        RT2 = np.count_nonzero(~RT2)
        if RT2 < 20.0:
            RT2 = 0

        # Array mit Daten für Sitzung 3 erstellen
        RT3 = np.array(data_results["RT3"])
        RT3 = np.isnan(RT3)
        RT3 = np.count_nonzero(~RT3)
        if RT3 < 20.0:
            RT3 = 0

        # Array mit Anzahl abgeschlossener Versuche pro Sitzung zurückgeben
        return np.array(
            [
                RT1,
                RT2,
                RT3,
            ]
        )

    #############################################################################################################
    ####################### Summe abgeschlossener Versuche pro Sitzung ausgeben und speichern ###################
    #############################################################################################################

    def save_sum_completed_tasks(self):

        data_anz_tasks = np.full((self.nr_subjects, 2, 3), 0)

        # Schleife für alle 19 Probanden und DBS-ON/OFF Zustand
        for sub_id in range(self.nr_subjects - 1):
            for dbs_mode_idx, dbs_mode in enumerate(["ON", "OFF"]):

                # Daten laden in denen steht, ob ein Versuch abgeschlossen oder nicht beendet wurde
                data_results = self.load_completed_tasks(
                    sub_id=sub_id, dbs_mode=dbs_mode
                )

                # falls laden schief geht, wird mit nächster Person/ nächstem Zustand weiter gemacht
                if data_results is None:
                    continue

                # Funktion zum summieren der abgeschlossenen Versuche Pro Sitzung
                data_anz_tasks[sub_id, dbs_mode_idx] = self.sum_completed_tasks(
                    data_results
                )
                data_anz_tasks[18, :] = (
                    40.0  # in Studiendaten fehlt bei P18 RT1,RT2 und RT3 -> manuell ausgelesen
                )

        DataON = data_anz_tasks[:, 0]
        DataOFF = data_anz_tasks[:, 1]

        ###################### Array mit Anzahl abgeschlossener Versuche pro Sitzung ausgeben ###################

        print("DBS ON")
        print(DataON)
        print("")
        print("DBS OFF")
        print(DataOFF)
        print("")

        ########## Array mit Anzahl abgeschlossener Versuche pro Sitzung in Excel Datei speichern ###############

        ## save data ON
        df = pd.DataFrame(data_anz_tasks[:, 0])
        filepath = "data/patient_data/Anz_CompleteTasks_ON.json"
        df.to_json(filepath, orient="records", lines=True)

        ## save data OFF
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

        ### print Daten
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

    data.save_sum_completed_tasks()
    data.save_sum_rewards()
    data.choices_rewards_per_trial()
