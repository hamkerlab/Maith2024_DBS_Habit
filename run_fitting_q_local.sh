#!/bin/bash

# temporarely disabled running patients models (data already fitted) and their comparison
# also temporarely disabled analysis of p explore because currently not updated for patients + simulations

# # Run single and double models with patients
# PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_single'  python fitting_q_learning.py single &
# pid1=$!
# PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_double'  python fitting_q_learning.py double &
# pid2=$!
# Run double model with simulations
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_suppression'  python fitting_q_learning.py suppression &
pid3=$!
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_efferent'  python fitting_q_learning.py efferent &
pid4=$!
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_dbs-all'  python fitting_q_learning.py dbs-all &
pid5=$!

# Wait for the scripts to finish
# wait $pid1
# wait $pid2
wait $pid3
wait $pid4
wait $pid5

# # Run comparison
# PYTENSOR_FLAGS='base_compiledir=./pytensor_compile'  python fitting_q_learning.py comparison

# get p explore estimates
python fitting_q_learning.py get_explore

# # analyze p explore
# python fitting_q_learning.py analyze_explore