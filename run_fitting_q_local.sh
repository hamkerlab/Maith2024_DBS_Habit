PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_single'  python fitting_q_learning.py single &
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile_double'  python fitting_q_learning.py double
PYTENSOR_FLAGS='base_compiledir=./pytensor_compile'  python fitting_q_learning.py comparison