# Maith2024_DBS_Habit
Source code of simulations and analyses from Maith, O., Apenburg, D., & Hamker, F. H. (2024). Pallidal deep brain stimulation enhances habitual behavior in a neuro-computational basal ganglia model during a reward reversal learning task. Submitted to European Journal of Neuroscience.

## Authors:

* Oliver Maith (oliver.maith@informatik.tu-chemnitz.de)
* Dave Apenburg (dave.apenburg@s2020.tu-chemnitz.de)

## Using the Scripts

**Generating Data:**
The data generated and analyzed in the study can also be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.12819011).

1. Download experimental data from De A Marcelino et al. (2023) and extract it into the "Data_experimental_study" folder. Download link: https://osf.io/fs36g/files/osfstorage (only the folder "Behavioural data" required). After extracting, the data files should be in the directory "Data_experimental_study/osfstorage-archive/Behavioral data/".
2. Rename the directory "Data_experimental_study/osfstorage-archive/Behavioral data" to "Behavioral_data". There should be no spaces in the path.
3. Run the "patient_data.py" script to process the experimental data.
4. In "run_simulation.py", set the variable `get_simulation_data` to True (set all other variables to False) and run the script. -> 1200 simulations
5. In "run_simulation.py", set the variable `get_activity_change_data` to True (set all other variables to False) and run the script. -> 1200 simulations (only first trial)
6. In "run_simulation.py", set the variable `get_dbs_parameter_data` to True (set all other variables to False) and run the script. -> 3752 simulations
7. In "run_simulation.py", set the variable `get_load_simulate_data` to True (set all other variables to False) and run the script. -> 2400 simulations + 2400 simulations (only third session)

**Generating Figures:**

In "run_simulation.py", set the variable `plot_figures` to True (set the variables for data generation to False) and run the script.

**Generating Statistical Results:**

In "run_simulation.py", set the variable `run_statistic` to True (set the variables for data generation and figure generation to False) and run the script.

  - All data, figures, and statistical analyses will be saved and retrieved in the existing folder structure. Placeholders are stored in this structure.

### Folders
**data:**
  - This folder stores all the data required to create the figures and results
  - Contains subfolders for simulation data, patient data, activity change data, parameter data, GPi scatter data

**Data_experimental_study:**
  - Folder for experimental data from De A Marcelino et al. (2023)
  - Download these data from the following and extract them into the "Data_experimental_study" folder. link: https://osf.io/fs36g/files/osfstorage

**fig:**
  - Placeholder ensures folder structure for figures

**statistic:**
  - Placeholder ensures folder structure for statistical tables

### Scripts

**BG_Model.py**
  - Basal ganglia model with all equations and connections of neurons and synapses

**parameters.py**
  - Parameters for the script "BG_Model.py"

**run_simulation.py**
  - Main script for generating data and creating figures and statistical tables

**patient_data.py**
  - Processes the patient data from De A Marcelino et al. (2023) and saves it
  - Creates a file for the number of completed trials and the sum of rewarded trials for both dbs-off and dbs-on

**simulation.py**
  - Started by "run_simulation.py"
  - Generates and saves simulation data, parameter data, and GPi scatter data

**simulation_activity_change.py**
  - Started by "run_simulation.py"
  - Generates and saves activity change data

**load_simulation.py**
  - Started by "run_simulation.py"
  - Generates and saves load simulation data

**visualization.py**
  - Started by "run_simulation.py"
  - Generates and saves all figures selected in "run_simulation.py"

**statistic.py**
  - Started by "run_simulation.py"
  - Generates and saves all statistical analyses selected in "run_simulation.py"
  - Contains helper functions for "visualization.py" and "run_simulation.py"


### Results Pipelines

Description of the results generated in Python. For several figures, additional image processing software was used to create the final figures (to adjust the layout and/or colors).

| Results        | Value in "run_simulation.py" | experimental data | simulated data | activity change data | parameter data | gpi scatter data | load simulation data | data script                       |
|----------------|------------------------------|-------------------|----------------|----------------------|----------------|------------------|---------------------|-----------------------------------|
| Figure 2       | **fig_shortcut_on_off**      | yes               | yes            | no                   | no             | no               | no                  | **simulation.py**                 |
| Figure 3       | **fig_dbs_on_off_14_100**    | yes               | yes            | no                   | no             | no               | no                  | **simulation.py**                 |
| Figure 4       | **fig_activity_changes_dbs_on** | no               | no             | yes                  | no             | no               | no                  | **simulation_activity_change.py** |
| Figure 5       | **fig_activity_changes_dbs_off** | no               | no             | yes                  | no             | no               | no                  | **simulation_activity_change.py** |
| Figure 6       | **fig_gpi_scatter**          | no                | no             | no                   | yes            | yes              | no                  | **simulation.py**                 |
| Figure 7       | **fig_load_simulate**        | no                | no             | no                   | no             | no               | yes                 | **load_simulation.py**            |
| Figure 8       | **fig_dbs_parameter**        | no                | no             | no                   | yes            | no               | no                  | **simulation.py**                 |
| Figure 9       | **fig_parameter_gpi_inhib**  | no                | no             | no                   | yes            | yes              | no                  | **simulation.py**                 |

All statistical analyses are started in **run_simulation.py** by setting the variable **run_statistic** to True. The variables for data collection and figure creation must be set to False. When the script is run, the statistical results are calculated in **statistic.py** and stored in the **statistics** folder.

# Platforms

* GNU/Linux

# Dependencies (given versions used in study)
For ANNarchy and CompNeuroPy the commits for the source code used for installation are given.

* Python >= 3.10.9
* ANNarchy = 4.7.3b ([1c60b7d095389ca521a8ef614624e7d90423f968](https://github.com/ANNarchy/ANNarchy/commit/1c60b7d095389ca521a8ef614624e7d90423f968))
* CompNeuroPy = 0.1.0 ([7c4f9dcb391e8408cafb340620982414b32a1269](https://github.com/Olimaol/CompNeuroPy/commit/7c4f9dcb391e8408cafb340620982414b32a1269))
* matplotlib >= 3.7.0
* numpy >= 1.23.5
* openpyxl >= 3.1.5
* pandas >= 1.5.3
* pingouin >= 0.5.4
* scipy >= 1.10.0

Note: In the versions of the packages and code being used, the variable **dbs_depolarization** from CompNeuroPy is incorrectly named and actually represents the variable for the *suppression* DBS variant.
