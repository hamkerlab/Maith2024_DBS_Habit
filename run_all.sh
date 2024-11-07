#!/bin/bash
# run it via ./run_all.sh
# if not possible make it executable: chmod +x run_all.sh
# takes one arguments
# first: how many cores should be used for running simulations in parallel

# Check if the argument was provided
if [ -z "$1" ]; then
  echo "Error: Missing argument (number of cores for running simulations in parallel)."
  echo "Usage: $0 <argument>"
  exit 1
fi

# run patient_data.py
python patient_data.py
# run run_simulation using given cores in mode 0 --> get_simulation_data
python run_simulation.py ${1} 0
rm -rf annarchy_folders/
# run run_simulation using given cores in mode 1 --> get_activity_change_data
python run_simulation.py ${1} 1
rm -rf annarchy_folders/
# run run_simulation using given cores in mode 2 --> get_dbs_parameter_data
python run_simulation.py ${1} 2
rm -rf annarchy_folders/
# run run_simulation using given cores in mode 3 --> get_load_simulate_data
python run_simulation.py ${1} 3
rm -rf annarchy_folders/
# run run_simulation using given cores in mode 4 --> run_statistic
python run_simulation.py ${1} 4
rm -rf annarchy_folders/
# run run_simulation using given cores in mode 5 --> plot_figures
python run_simulation.py ${1} 5
rm -rf annarchy_folders/