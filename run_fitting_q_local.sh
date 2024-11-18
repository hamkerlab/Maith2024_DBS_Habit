#!/bin/bash

# Run single and double models
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_single'  python fitting_q_learning.py single &
pid1=$!
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_double'  python fitting_q_learning.py double &
pid2=$!

# Wait for both scripts to finish
wait $pid1
wait $pid2

# Run comparison
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile'  python fitting_q_learning.py comparison

# get p explore estimates
python fitting_q_learning.py get_explore

# analyze p explore
python fitting_q_learning.py analyze_explore