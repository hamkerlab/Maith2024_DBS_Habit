from BG_model import *
import ANNarchy as ann

params = {}
# params['num_threads'] = 4

##############################################################################
############################ parameters neurons ##############################
##############################################################################

########################## geometry ##########################################
params["dim_IT"] = 2
params["dim_STN"] = 4
params["dim_STR"] = 4
params["dim_StrThal"] = 2
params["dim_SNc"] = 1
params["dim_PPN"] = 1
params["dim_GPi"] = 2
params["dim_GPe"] = 2
params["dim_Thal"] = 2
params["dim_PFC"] = 2

########################## Baseline ##########################################
params["baseline_IT"] = 0.0
params["baseline_dopa"] = 0.1  # 0.1
params["baseline_StrD1"] = 0.1  # 0.1
params["baseline_StrD2"] = 0.1  # 0.1
params["baseline_SNc"] = 0.1  # 0.1
params["baseline_StrThal"] = 0.25  # 0.4 #DBS0.25
params["baseline_GPi"] = 2.1
params["baseline_STN"] = 0.1  # 0.1
params["baseline_GPe"] = 1.0
params["baseline_Thal"] = 1.0
params["baseline_PFC"] = 0.0

########################### Noise ############################################
params["noise_Str"] = 0.1
params["noise_StrThal"] = 0.1
params["noise_GPi"] = 0.1  # 0.1
params["noise_STN"] = 0.1
params["noise_GPe"] = 0.1
params["noise_Thal"] = 0.0001
params["noise_PFC"] = 0.05  # 0.05
params["noise_IT"] = 0.0

##############################################################################
############################ parameter synapses ##############################
##############################################################################

############################# mean input #####################################

params["Mittelwert_umschalten"] = False

################################## Input #####################################

params["ITStrD1_regularization_threshold"] = 0.7
params["ITStrD1_tau"] = 100.0  # 100.0 #DBS140
params["ITStrD1_K_burst"] = 2.0  # 2.0
params["ITStrD1_K_dip"] = 0.4  # 0.4
params["ITStrD1_threshold_pre"] = 1.0  # 1.0

params["ITStrD2_regularization_threshold"] = 1.0
params["ITStrD2_tau"] = 70.0  # 70.0 #DBS150
params["ITStrD2_K_burst"] = 0.6  # 0.5#
params["ITStrD2_K_dip"] = 0.4  # 0.4
params["ITStrD2_threshold_pre"] = 1.0  # 1.0

params["ITSTN_regularization_threshold"] = 0.7
params["ITSTN_tau"] = 70.0  # 70.0 #DBS150
params["ITSTN_K_burst"] = 1.0  # 1.0
params["ITSTN_K_dip"] = 0.4  # 0.4
params["ITSTN_threshold_pre"] = 1.0  # 1.0

################################ Inner BG ###################################
params["StrD1GPi_threshold_post"] = 0.10  # 0.10
params["StrD1GPi_trace_neg_factor"] = 0.75  # 0.75
params["StrD1GPi_regularization_threshold"] = 1.0  # 1.0
params["StrD1GPi_tau"] = 50  # 50 #DBS100
params["StrD1GPi_K_burst"] = 1.5  # 1.0
params["StrD1GPi_K_dip"] = 0.4  # 0.4

params["StrD2GPe_threshold_post"] = 0.15  # 0.15
params["StrD2GPe_trace_neg_factor"] = 0.1  # 0.1
params["StrD2GPe_regularization_threshold"] = 2.0  # 2.0
params["StrD2GPe_tau"] = 60  # 60 #DBS100
params["StrD2GPe_K_burst"] = 2.0  # 2.0
params["StrD2GPe_K_dip"] = 0.4  # 0.4

params["STNGPi_threshold_post"] = 0.15  # 0.15
params["STNGPi_trace_neg_factor"] = 1.0
params["STNGPi_regularization_threshold"] = 1.0
params["STNGPi_tau"] = 50  # 50 #DBS100
params["STNGPi_K_burst"] = 1.0
params["STNGPi_K_dip"] = 0.4

##############################################################################
################################ weigths #####################################
##############################################################################

############################ lateral weigths ###############################
params["StrD1StrD1.connect_all_to_all"] = 1.0  # 1.0
params["StrD2StrD2.connect_all_to_all"] = 1.2  # 1.2
params["STNSTN.connect_all_to_all"] = 1.0  # 1.0
params["GPiGPi.connect_all_to_all"] = 0.9  # 0.9
params["StrThalStrThal.connect_all_to_all"] = 1.0  # 1.0

############################ thalamus feedback ################################
params["ThalStrThal.connect_one_to_one"] = 0.5  # 0.5   #statt 1.00
params["StrThalGPi.connect_one_to_one"] = 0.5  # 0.5    #statt 0.3
params["StrThalGPe.connect_one_to_one"] = 0.3  # 0.3

################################## inner BG ###################################
params["StrD1GPi.connect_all_to_all"] = ann.Uniform(0.0, 0.05)
params["StrD2GPe.connect_all_to_all"] = 0.0  # 0.0
params["GPeGPi.connect_one_to_one"] = 1.5  # 1.5
params["GPeSTN.connect_all_to_all"] = 0.1
params["STNGPe.connect_all_to_all"] = 0.1

################################## output #####################################
params["GPiThal.connect_one_to_one"] = 1.0  # 0.75
params["ThalPFC.connect_one_to_one"] = 1.0

###############################################################################
################################ shortcut #####################################
###############################################################################

params["ITThal_tau"] = 200000  # 150000
params["ITThal_tau_alpha"] = 1.0  # 1.0
params["ITThal_regularization_threshold"] = 0.93  # 0.93
params["ITThal_threshold_post"] = 0.1  # 0.1
params["ITThal_threshold_pre"] = 0.0  # 0.1
params["ITThal_alpha_factor"] = 2.0  # 2.0
# weigths
params["ITThal.connect_all_to_all"] = 0.1  # 0.1

###############################################################################
################################## DBS ########################################
###############################################################################

############################ 1 - suppression ##############################
# 0.1 - 0.25
params["1_suppression"] = 0.1  # 0.1
params["1_frequence"] = 130
params["1_pulse_width"] = 10000

############################ 2 - efferent axons ##############################
# 0.0 - 0.05
params["2_suppression"] = 0.0
params["2_frequence"] = 130
params["2_pulse_width"] = 10000
params["2_axon_rate_amplitude"] = 0.05  # 0.05

############################ 3 - afferent axons ##############################
# 0.14 - 0.20
params["3_suppression"] = 0.0
params["3_frequence"] = 130
params["3_pulse_width"] = 10000
params["3_axon_rate_amplitude"] = 0.05  # 0.05

######################### 4 - passing fibres ##################################
# Auswirkungen zwischen 0.4 und 0.5
params["4_suppression"] = 0.0
params["4_frequence"] = 130
params["4_pulse_width"] = 10000
params["4_axon_rate_amplitude"] = 0.05  # 0.05
params["4_fibre_strength"] = 7.5  # 7.5

############################ 5 - dbs all ######################################
params["5_suppression"] = 0.1  # 0.1
params["5_frequence"] = 130
params["5_pulse_width"] = 10000
params["5_axon_rate_amplitude"] = 0.03  # 0.03
params["5_fibre_strength"] = 7.5  # 7.5
