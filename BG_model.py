"""
Basal Ganglia Model based on the version of F. Escudero
Modified by A. Schwarz
Version 1.0 - 29.05.2018
"""

# from ANNarchy import *
from ANNarchy import (
    Constant,
    Neuron,
    Synapse,
    Population,
    Projection,
    get_population,
    get_projection,
    Uniform,
    Monitor,
    compile,
    setup,
    reset,
    clear,
)
from parameters import params
import numpy as np
from time import time
from CompNeuroPy import DBSstimulator

############################# clear network ##################################
clear()

##############################################################################
################################# init variables #############################
##############################################################################

baseline_dopa = Constant("baseline_dopa", params["baseline_dopa"])


def create_network(seed, dbs_state, shortcut, parameter, dbs_param_state):
    ######################### Setup ANNarchy function ############################

    # load seed parameters
    setup(seed=seed)

    ##############################################################################
    ################################# neurons ####################################
    ##############################################################################

    ################################# input ######################################

    # PPN, IT
    InputNeuron = Neuron(
        parameters="""
            tau = 10.0 : population
            B = 0.0
        """,
        equations="""
            tau * dmp/dt = -mp + B
            r = pos(mp)
        """,
        name="Input Neuron",
        description="Rate-coded neuron only with a baseline to be set.",
    )

    ################################# standard ####################################
    # StrD1, StrD2, STN, GPi, GPe, Thal, PFC, StrThal
    LinearNeuron = Neuron(
        parameters="""
            tau = 10.0 : population
            phi = 0.0 : population
            B = 0.0
            input = 0.0
        """,
        equations="""
            tau * dmp/dt = -mp + sum(exc) - sum(inh) + B + input + phi * Uniform(-1.0,1.0)
            r = pos(mp)
        """,
        name="Linear Neuron",
        description="Regular rate-coded neuron with excitatory and inhibitory inputs plus baseline and noise.",
    )

    ################################# Dopamin ######################################
    # SNc
    DopamineNeuron = Neuron(
        parameters="""
            tau = 10.0 : population
            alpha = 0 : population
            B = 0.0
        """,
        equations="""
            aux = if (sum(exc)>0): pos(1.0-B-sum(inh)) else: -10 * sum(inh)
            tau * dmp/dt = -mp + alpha * aux + B
            r = pos(mp)
        """,
        name="Dopamine Neuron",
        description="Excitatory input increases activity above constant baseline. Inhibitory input can prevent the increase.",
    )

    ##############################################################################
    ################################# synapses ###################################
    ##############################################################################

    ########################## standard synapses (DA) #############################
    # PPNSNc, SNcStrD1, SNcStrD2, SNcSTN, SNcGPe, SNcGPi
    StandardSynapse = Synapse(
        psp="w * pre.r",
        name="Standard",
        description="Standard synapse, without plasticity which calculates the psp as a multiplication of weight and pre-synaptic rate.",
    )

    # ITThal
    PostCovariance = Synapse(
        parameters="""
            tau = 15000.0 : projection
            tau_alpha = 1.0 : projection
            regularization_threshold = 0.5: projection
            threshold_post = 0.1 : projection
            threshold_pre = 0.0 : projection
            alpha_factor = 2.0 : projection
        """,
        psp="w * pre.r",
        equations="""
            tau_alpha * dalpha/dt  + alpha =  pos(post.mp - regularization_threshold) * alpha_factor
            trace = pos(pre.r - threshold_pre) * (post.r - mean(post.r) - threshold_post)
            delta = (trace - alpha*pos(post.r - mean(post.r) - threshold_post) * pos(post.r - mean(post.r) - threshold_post)*w)
            tau * dw/dt = delta : min = 0
    """,
        name="Covariance learning rule",
        description="Synaptic plasticity based on covariance, with an additional regularization term.",
    )

    ############################## striatal synapse ##################################
    # ITStrD1, ITStrD2, ITSTN
    DAPostCovarianceNoThreshold = Synapse(
        parameters="""
            tau=75.0 : projection
            tau_alpha=1.0 : projection
            tau_trace=60.0 : projection
            regularization_threshold=1.0 : projection
            K_burst = 1.0 : projection
            K_dip = 0.4 : projection
            DA_type = 1 : projection
            threshold_pre=0.15 : projection
            threshold_post=0.0 : projection
        """,
        psp="w * pre.r",
        equations="""
            tau_alpha * dalpha/dt + alpha = pos(post.mp - regularization_threshold)
            dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)
            trace = pos(post.r -  mean(post.r) - threshold_post) * pre.r 
            
            dopa_mod =  if (DA_type*dopa_sum>0): 
                            DA_type*K_burst*dopa_sum 
                        else: 
                            if (trace>0.0) and (w >0.0):
                                DA_type*K_dip*dopa_sum
                            else:
                                0
            
            delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
            
            tau * dw/dt = delta : min=0
        """,
        name="Covariance DA learning rule",
        description="Synaptic plasticity in the BG input, like the Covariance learning rule with an additional dopamine modulation which depends on the dopamine receptor type ($DA_{type}(D1) = 1, DA_{type}(D2) = -1$).",
    )

    ############################## pallidal synapse (exc) ##################################
    # STNGPi
    DA_excitatory = Synapse(
        parameters="""
            tau=50.0 : projection
            tau_alpha=1.0 : projection
            tau_trace=60.0 : projection
            regularization_threshold=2.6 : projection
            K_burst = 1.0 : projection
            K_dip = 0.4 : projection
            DA_type= 1 : projection
            threshold_pre=0.0 : projection
            threshold_post= -0.15 : projection
            trace_pos_factor = 1.0 : projection
            gamma = 0.45 : projection
        """,
        psp="w * pre.r",
        equations="""
            tau_alpha * dalpha/dt + alpha = pos(post.mp - regularization_threshold)
            dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)

            a = mean(post.r) - min(post.r) - gamma : postsynaptic
            post_thresh =   if (-a<threshold_post): 
                                -a 
                            else: 
                                threshold_post : postsynaptic

            trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - post_thresh)
            aux = if (trace<0.0): 1 else: 0
            dopa_mod =  if (dopa_sum>0): 
                            K_burst * dopa_sum * ((1-trace_pos_factor)*aux+trace_pos_factor) 
                        else: 
                            K_dip * dopa_sum * aux
            delta = dopa_mod * trace - alpha * pos(trace)
            tau * dw/dt = delta : min=0
        """,
        name="STN Output learning rule",
        description="Synaptic plasticity in the STN output, similar to the Covariance learning rule with an additional dopamine modulation.",
    )

    ############################## pallidal synapse (inh) ##############################
    # StrD1GPi, StrD2GPe
    DA_inhibitory = Synapse(
        parameters="""
            tau=50.0 : projection
            tau_alpha=1.0 : projection
            tau_trace=60.0 : projection
            regularization_threshold=1.0 : projection
            K_burst = 1.0 : projection
            K_dip = 0.4 : projection
            DA_type= 1 : projection
            threshold_pre=0.0 : projection
            threshold_post=0.15 : projectionbaseline
            trace_neg_factor = 1.0 : projection
            gamma = 0.45 : projection
        """,
        psp="w * pre.r",
        equations="""
            tau_alpha * dalpha/dt + alpha = pos(-post.mp - regularization_threshold)
            dopa_sum = 2.0 * (post.sum(dopa) - baseline_dopa)

            a = mean(post.r) - min(post.r) - gamma : postsynaptic
            post_thresh =   if a>threshold_post: 
                                if DA_type>0: 
                                    a
                                else:
                                    threshold_post : postsynaptic
                            else: 
                                threshold_post : postsynaptic

            trace =     if (DA_type>0): 
                            pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - post_thresh) 
                        else: 
                            pos(pre.r - mean(pre.r) - threshold_pre) * (max(post.r) - post.r  - post_thresh)
            
            aux = if (trace>0): 1 else: 0
            
            dopa_mod =  if (DA_type*dopa_sum>0): 
                            DA_type*K_burst*dopa_sum * ((1-trace_neg_factor)*aux+trace_neg_factor) 
                        else: 
                            aux*DA_type*K_dip*dopa_sum
                            
            tau * dw/dt = dopa_mod * trace - alpha * pos(trace) : min=0
        """,
        name="Str Output learning rule",
        description="Synaptic plasticity in the Str output, similar to the Covariance learning (here inverse effect of post-activity) rule with an additional dopamine modulation which depends on the dopamine receptor type ($DA_{type}(D1) = 1, DA_{type}(D2) = -1$).",
    )

    ########################## dopamin synapse ##################################
    # StrD1SNc
    DAPrediction = Synapse(
        parameters="""
            tau = 300.0 : projection
        """,
        psp="w * pre.r",
        equations="""
            aux = if (post.sum(exc)>0): 1.0 else: 3.0  : postsynaptic
            tau*dw/dt = aux * (post.r - baseline_dopa) * pos(pre.r - mean(pre.r)) : min=0
        """,
        name="Reward Prediction learning rule",
        description="Simple synaptic plasticity based on covariance.",
    )

    ##############################################################################
    ################################# population #################################
    ##############################################################################

    # IT Input
    IT = Population(name="IT", geometry=params["dim_IT"], neuron=InputNeuron)
    IT.B = params["baseline_IT"]
    IT.phi = params["noise_IT"]

    # Reward Input
    PPN = Population(name="PPN", geometry=params["dim_PPN"], neuron=InputNeuron)
    PPN.tau = 1.0

    # PFC
    PFC = Population(name="PFC", geometry=params["dim_PFC"], neuron=LinearNeuron)
    PFC.phi = params["noise_PFC"]
    PFC.B = params["baseline_PFC"]  # pfc_base

    # SNc
    SNc = Population(name="SNc", geometry=params["dim_SNc"], neuron=DopamineNeuron)
    SNc.B = params["baseline_SNc"]

    # Striatum direct pathway
    StrD1 = Population(name="StrD1", geometry=params["dim_STR"], neuron=LinearNeuron)
    StrD1.phi = params["noise_Str"]
    StrD1.B = params["baseline_StrD1"]

    # Striatum indirect pathway
    StrD2 = Population(name="StrD2", geometry=params["dim_STR"], neuron=LinearNeuron)
    StrD2.phi = params["noise_Str"]
    StrD2.B = params["baseline_StrD2"]

    # Striatum feedback pathway
    StrThal = Population(
        name="StrThal", geometry=params["dim_StrThal"], neuron=LinearNeuron
    )
    StrThal.phi = params["noise_StrThal"]
    StrThal.B = params["baseline_StrThal"]

    # GPi
    GPi = Population(name="GPi", geometry=params["dim_GPi"], neuron=LinearNeuron)
    GPi.phi = params["noise_GPi"]
    GPi.B = params["baseline_GPi"]

    # STN
    STN = Population(name="STN", geometry=params["dim_STN"], neuron=LinearNeuron)
    STN.phi = params["noise_STN"]
    STN.B = params["baseline_STN"]

    # GPe
    GPe = Population(name="GPe", geometry=params["dim_GPe"], neuron=LinearNeuron)
    GPe.phi = params["noise_GPe"]
    GPe.B = params["baseline_GPe"]

    # Thal
    Thal = Population(name="Thal", geometry=params["dim_Thal"], neuron=LinearNeuron)
    Thal.phi = params["noise_Thal"]
    Thal.B = params["baseline_Thal"]

    ##############################################################################
    ################################# projektion #################################
    ##############################################################################

    ################################# Input ######################################
    # shortcut IT -> Thalamus (plastic or fixed))
    if shortcut == False:
        ITThal = Projection(
            pre=IT, post=Thal, target="exc", synapse=StandardSynapse, name="ITThal"
        )
        ITThal.connect_all_to_all(
            weights=params["ITThal.connect_all_to_all"]
        )  # Normal(0.3,0.1) )

    if shortcut == True:
        ITThal = Projection(
            pre=IT, post=Thal, target="exc", synapse=PostCovariance, name="ITThal"
        )
        ITThal.connect_all_to_all(weights=params["ITThal.connect_all_to_all"])
        ITThal.tau = params["ITThal_tau"]
        ITThal.tau_alpha = params["ITThal_tau_alpha"]
        ITThal.regularization_threshold = params["ITThal_regularization_threshold"]
        ITThal.threshold_post = params["ITThal_threshold_post"]
        ITThal.threshold_pre = params["ITThal_threshold_pre"]
        ITThal.alpha_factor = params["ITThal_alpha_factor"]

    ITStrD1 = Projection(
        pre=IT,
        post=StrD1,
        target="exc",
        synapse=DAPostCovarianceNoThreshold,
        name="ITStrD1",
    )
    ITStrD1.connect_all_to_all(weights=Uniform(0, 0.1))  # Normal(0.15,0.15))
    ITStrD1.regularization_threshold = params["ITStrD1_regularization_threshold"]
    ITStrD1.tau = params["ITStrD1_tau"]
    ITStrD1.K_burst = params["ITStrD1_K_burst"]
    ITStrD1.K_dip = params["ITStrD1_K_dip"]

    ITStrD2 = Projection(
        pre=IT,
        post=StrD2,
        target="exc",
        synapse=DAPostCovarianceNoThreshold,
        name="ITStrD2",
    )
    ITStrD2.connect_all_to_all(weights=Uniform(0, 0.1))  # Normal(0.15,0.15))
    ITStrD2.DA_type = -1
    ITStrD2.regularization_threshold = params["ITStrD2_regularization_threshold"]
    ITStrD2.tau = params["ITStrD2_tau"]
    ITStrD2.K_burst = params["ITStrD2_K_burst"]
    ITStrD2.K_dip = params["ITStrD2_K_dip"]

    ITSTN = Projection(
        pre=IT,
        post=STN,
        target="exc",
        synapse=DAPostCovarianceNoThreshold,
        name="ITSTN",
    )
    ITSTN.connect_all_to_all(weights=Uniform(0, 0.1))  # Normal(0.15,0.15))
    ITSTN.DA_type = 1
    ITSTN.regularization_threshold = params["ITSTN_regularization_threshold"]
    ITSTN.tau = params["ITSTN_tau"]
    ITSTN.K_burst = params["ITSTN_K_burst"]
    ITSTN.K_dip = params["ITSTN_K_dip"]

    ########################## output ##################################

    GPiThal = Projection(pre=GPi, post=Thal, target="inh", synapse=StandardSynapse)
    GPiThal.connect_one_to_one(weights=params["GPiThal.connect_one_to_one"])

    ThalPFC = Projection(
        pre=Thal, post=PFC, target="exc", synapse=StandardSynapse, name="ThalPFC"
    )
    ThalPFC.connect_one_to_one(weights=params["ThalPFC.connect_one_to_one"])  # init

    ########################## dopamin ##################################

    PPNSNc = Projection(
        pre=PPN, post=SNc, target="exc", synapse=StandardSynapse, name="PPNSNc"
    )
    PPNSNc.connect_all_to_all(weights=1.0)

    StrD1SNc = Projection(
        pre=StrD1, post=SNc, target="inh", synapse=DAPrediction, name="StrD1SNc"
    )
    StrD1SNc.connect_all_to_all(weights=0.1)  # 0.5

    SNcStrD1 = Projection(
        pre=SNc, post=StrD1, target="dopa", synapse=StandardSynapse, name="SNcStrD1"
    )
    SNcStrD1.connect_all_to_all(weights=1.0)  # 1.0

    SNcStrD2 = Projection(
        pre=SNc, post=StrD2, target="dopa", synapse=StandardSynapse, name="SNcStrD2"
    )
    SNcStrD2.connect_all_to_all(weights=1.0)

    SNcGPi = Projection(
        pre=SNc, post=GPi, target="dopa", synapse=StandardSynapse, name="SNcGPi"
    )
    SNcGPi.connect_all_to_all(weights=1.0)

    SNcSTN = Projection(
        pre=SNc, post=STN, target="dopa", synapse=StandardSynapse, name="SNcSTN"
    )
    SNcSTN.connect_all_to_all(weights=1.0)

    SNcGPe = Projection(
        pre=SNc, post=GPe, target="dopa", synapse=StandardSynapse, name="SNcGPe"
    )
    SNcGPe.connect_all_to_all(weights=1.0)

    ########################## Inner BG ######################################

    StrD1GPi = Projection(
        pre=StrD1, post=GPi, target="inh", synapse=DA_inhibitory, name="StrD1GPi"
    )
    StrD1GPi.connect_all_to_all(weights=Uniform(0.0, 0.05))  # Normal(0.025,0.025))
    StrD1GPi.DA_type = 1
    StrD1GPi.threshold_post = params["StrD1GPi_threshold_post"]
    StrD1GPi.trace_neg_factor = params["StrD1GPi_trace_neg_factor"]
    StrD1GPi.regularization_threshold = params["StrD1GPi_regularization_threshold"]
    StrD1GPi.tau = params["StrD1GPi_tau"]
    StrD1GPi.K_burst = params["StrD1GPi_K_burst"]
    StrD1GPi.K_dip = params["StrD1GPi_K_dip"]

    STNGPi = Projection(
        pre=STN, post=GPi, target="exc", synapse=DA_excitatory, name="STNGPi"
    )
    STNGPi.connect_all_to_all(weights=Uniform(0, 0.05))  # Normal(0.025,0.025))
    STNGPi.threshold_post = params["STNGPi_threshold_post"]
    STNGPi.trace_neg_factor = params["STNGPi_trace_neg_factor"]
    STNGPi.regularization_threshold = params["STNGPi_regularization_threshold"]
    STNGPi.tau = params["STNGPi_tau"]
    STNGPi.K_burst = params["STNGPi_K_burst"]
    STNGPi.K_dip = params["STNGPi_K_dip"]

    StrD2GPe = Projection(
        pre=StrD2, post=GPe, target="inh", synapse=DA_inhibitory, name="StrD2GPe"
    )
    StrD2GPe.connect_all_to_all(
        weights=params["StrD2GPe.connect_all_to_all"]
    )  # Normal(0.025,0.025))
    StrD2GPe.DA_type = -1
    StrD2GPe.threshold_post = params["StrD2GPe_threshold_post"]
    StrD2GPe.trace_neg_factor = params["StrD2GPe_trace_neg_factor"]
    StrD2GPe.regularization_threshold = params["StrD2GPe_regularization_threshold"]
    StrD2GPe.tau = params["StrD2GPe_tau"]
    StrD2GPe.K_burst = params["StrD2GPe_K_burst"]
    StrD2GPe.K_dip = params["StrD2GPe_K_dip"]

    GPeGPi = Projection(pre=GPe, post=GPi, target="inh", synapse=StandardSynapse)
    GPeGPi.connect_one_to_one(weights=params["GPeGPi.connect_one_to_one"])

    GPeSTN = Projection(pre=GPe, post=STN, target="inh", synapse=StandardSynapse)
    GPeSTN.connect_all_to_all(weights=params["GPeSTN.connect_all_to_all"])

    STNGPe = Projection(pre=STN, post=GPe, target="exc", synapse=StandardSynapse)
    STNGPe.connect_all_to_all(weights=params["STNGPe.connect_all_to_all"])

    ####################### lateral connection ##################################

    StrD1StrD1 = Projection(
        pre=StrD1, post=StrD1, target="inh", synapse=StandardSynapse
    )
    StrD1StrD1.connect_all_to_all(weights=params["StrD1StrD1.connect_all_to_all"])

    STNSTN = Projection(pre=STN, post=STN, target="inh", synapse=StandardSynapse)
    STNSTN.connect_all_to_all(weights=params["STNSTN.connect_all_to_all"])

    StrD2StrD2 = Projection(
        pre=StrD2, post=StrD2, target="inh", synapse=StandardSynapse
    )
    StrD2StrD2.connect_all_to_all(weights=params["StrD2StrD2.connect_all_to_all"])

    StrThalStrThal = Projection(
        pre=StrThal, post=StrThal, target="inh", synapse=StandardSynapse
    )
    StrThalStrThal.connect_all_to_all(
        weights=params["StrThalStrThal.connect_all_to_all"]
    )

    GPiGPi = Projection(pre=GPi, post=GPi, target="inh", synapse=StandardSynapse)
    GPiGPi.connect_all_to_all(weights=params["GPiGPi.connect_all_to_all"])

    ########################### Thalamus Feedback ################################

    ThalStrThal = Projection(
        pre=Thal, post=StrThal, target="exc", synapse=StandardSynapse
    )
    ThalStrThal.connect_one_to_one(weights=params["ThalStrThal.connect_one_to_one"])

    StrThalGPe = Projection(
        pre=StrThal, post=GPe, target="inh", synapse=StandardSynapse
    )
    StrThalGPe.connect_one_to_one(weights=params["StrThalGPe.connect_one_to_one"])

    StrThalGPi = Projection(
        pre=StrThal, post=GPi, target="inh", synapse=StandardSynapse
    )
    StrThalGPi.connect_one_to_one(weights=params["StrThalGPi.connect_one_to_one"])

    ##############################################################################
    ############################### DBS Stimulator ###############################
    ##############################################################################

    if dbs_param_state == 0:
        # dbs_state = 0 -> DBS-OFF
        if dbs_state == 0:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=0.0,
                dbs_pulse_frequency_Hz=0.0,
                dbs_pulse_width_us=0.0,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 1 -> supress lokal neurons
        if dbs_state == 1:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["1_suppression"],
                dbs_pulse_frequency_Hz=params["1_frequence"],
                dbs_pulse_width_us=params["1_pulse_width"],
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 2 -> activate efferent axons
        if dbs_state == 2:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["2_suppression"],
                dbs_pulse_frequency_Hz=params["2_frequence"],
                dbs_pulse_width_us=params["2_pulse_width"],
                axon_rate_amp=params["2_axon_rate_amplitude"],
                orthodromic=True,
                efferents=True,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 3 -> activate afferent axons
        if dbs_state == 3:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["3_suppression"],
                dbs_pulse_frequency_Hz=params["3_frequence"],
                dbs_pulse_width_us=params["3_pulse_width"],
                axon_rate_amp=params["3_axon_rate_amplitude"],
                orthodromic=True,
                afferents=True,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 4 -> activate passing neurons
        if dbs_state == 4:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["4_suppression"],
                dbs_pulse_frequency_Hz=params["4_frequence"],
                dbs_pulse_width_us=params["4_pulse_width"],
                orthodromic=True,
                passing_fibres=True,
                passing_fibres_list=[
                    GPeSTN,
                ],
                axon_rate_amp=params["4_axon_rate_amplitude"],
                passing_fibres_strength=params["4_fibre_strength"],
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 5 -> dbs-comb
        if dbs_state == 5:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["5_suppression"],
                dbs_pulse_frequency_Hz=params["5_frequence"],
                dbs_pulse_width_us=params["5_pulse_width"],
                axon_rate_amp=params["5_axon_rate_amplitude"],
                orthodromic=True,
                efferents=True,
                afferents=False,
                passing_fibres=True,
                passing_fibres_list=[
                    GPeSTN,
                ],
                passing_fibres_strength=params["5_fibre_strength"],
                seed=seed,
                auto_implement=True,
            )
    else:
        # dbs_state = 0 -> DBS-OFF
        if dbs_state == 0:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=0.0,
                dbs_pulse_frequency_Hz=0.0,
                dbs_pulse_width_us=0.0,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 1 -> supress lokal neurons
        if dbs_state == 1:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=parameter,
                dbs_pulse_frequency_Hz=params["1_frequence"],
                dbs_pulse_width_us=params["1_pulse_width"],
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 2 -> activate efferent axons
        if dbs_state == 2:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["2_suppression"],
                dbs_pulse_frequency_Hz=params["2_frequence"],
                dbs_pulse_width_us=params["2_pulse_width"],
                axon_rate_amp=parameter,
                orthodromic=True,
                efferents=True,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 3 -> activate afferent axons
        if dbs_state == 3:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["3_suppression"],
                dbs_pulse_frequency_Hz=params["3_frequence"],
                dbs_pulse_width_us=params["3_pulse_width"],
                axon_rate_amp=parameter,
                orthodromic=True,
                afferents=True,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 4 -> activate passing neurons
        if dbs_state == 4:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["4_suppression"],
                dbs_pulse_frequency_Hz=params["4_frequence"],
                dbs_pulse_width_us=params["4_pulse_width"],
                orthodromic=True,
                passing_fibres=True,
                passing_fibres_list=[
                    GPeSTN,
                ],
                axon_rate_amp=params["4_axon_rate_amplitude"],
                passing_fibres_strength=parameter,
                seed=seed,
                auto_implement=True,
            )

        # dbs_state = 5 -> dbs-comb
        if dbs_state == 5:
            dbs = DBSstimulator(
                stimulated_population=GPi,
                dbs_depolarization=params["5_suppression"],
                dbs_pulse_frequency_Hz=params["5_frequence"],
                dbs_pulse_width_us=params["5_pulse_width"],
                axon_rate_amp=params["5_axon_rate_amplitude"],
                orthodromic=True,
                efferents=True,
                afferents=False,
                passing_fibres=True,
                passing_fibres_list=[
                    GPeSTN,
                ],
                passing_fibres_strength=params["5_fibre_strength"],
                seed=seed,
                auto_implement=True,
            )

    ########################### set pointer on population #############################
    (
        IT,
        PFC,
        StrD1,
        StrD2,
        STN,
        GPe,
        GPi,
        Thal,
        SNc,
        StrThal,
        PPN,
        PFC,
        ITStrD1,
        ITStrD2,
        ITSTN,
        StrD1GPi,
        StrD2GPe,
        STNGPi,
        StrD1SNc,
        ITThal,
        ThalPFC,
    ) = dbs.update_pointers(
        [
            IT,
            PFC,
            StrD1,
            StrD2,
            STN,
            GPe,
            GPi,
            Thal,
            SNc,
            StrThal,
            PPN,
            PFC,
            ITStrD1,
            ITStrD2,
            ITSTN,
            StrD1GPi,
            StrD2GPe,
            STNGPi,
            StrD1SNc,
            ITThal,
            ThalPFC,
        ]
    )

    ########################### compile network #############################
    compile()

    populations = [
        IT,
        PFC,
        StrD1,
        StrD2,
        STN,
        GPe,
        GPi,
        Thal,
        SNc,
        StrThal,
        PPN,
        PFC,
        ITStrD1,
        ITStrD2,
        ITSTN,
        StrD1GPi,
        StrD2GPe,
        STNGPi,
        StrD1SNc,
        ITThal,
        ThalPFC,
        dbs,
    ]

    return populations


##############################################################################
################################# Monitoring #################################
##############################################################################

########################### record firing rates ##############################


class BGMonitor(object):
    def __init__(self, populations):
        self.populations = populations
        self.monitors = []
        for pop in populations:
            self.monitors.append(Monitor(pop, "r", start=False))

    def start(self):
        for monitor in self.monitors:
            monitor.start()

    def stop(self):
        for monitor in self.monitors:
            monitor.pause()

    def get(self):
        res = {}
        for monitor in self.monitors:
            res[monitor.object.name] = monitor.get("r")
        return res


############################### record weigths ###############################


def extract_data(
    SNc,
    ITStrD1,
    ITStrD2,
    ITSTN,
    StrD1GPi,
    StrD2GPe,
    STNGPi,
    StrD1SNc,
    ITThal,
    ThalPFC,
    shortcut,
):

    # 0
    IT_StrD1 = np.array([dendrite.w for dendrite in ITStrD1])
    IT_StrD1[IT_StrD1 < 0] = 0.0  # Alles kleiner 0 wird durch 0 ersetzt
    IT_StrD1 = np.reshape(
        IT_StrD1, (params["dim_IT"] * params["dim_STR"], 1)
    )  # Array in 8x1 Dimension wandeln
    IT_StrD1 = IT_StrD1.tolist()  # Array zu Liste aus Arrays wandeln
    # 1
    IT_StrD2 = np.array([dendrite.w for dendrite in ITStrD2])
    IT_StrD2[IT_StrD2 < 0] = 0.0
    IT_StrD2 = np.reshape(IT_StrD2, (params["dim_IT"] * params["dim_STR"], 1))
    IT_StrD2 = IT_StrD2.tolist()
    # 2
    IT_STN = np.array([dendrite.w for dendrite in ITSTN])
    IT_STN[IT_STN < 0] = 0.0
    IT_STN = np.reshape(IT_STN, (params["dim_IT"] * params["dim_STN"], 1))
    IT_STN = IT_STN.tolist()
    # 3
    StrD1_GPi = np.array([dendrite.w for dendrite in StrD1GPi])
    StrD1_GPi[StrD1_GPi < 0] = 0.0
    StrD1_GPi = np.reshape(StrD1_GPi, (params["dim_GPi"] * params["dim_STR"], 1))
    StrD1_GPi = StrD1_GPi.tolist()
    # 4
    StrD2_GPe = np.array([dendrite.w for dendrite in StrD2GPe])
    StrD2_GPe[StrD2_GPe < 0] = 0.0
    StrD2_GPe = np.reshape(StrD2_GPe, (params["dim_GPe"] * params["dim_STR"], 1))
    StrD2_GPe = StrD2_GPe.tolist()
    # 5
    STN_GPi = np.array([dendrite.w for dendrite in STNGPi])
    STN_GPi[STN_GPi < 0] = 0.0
    STN_GPi = np.reshape(STN_GPi, (params["dim_GPi"] * params["dim_STN"], 1))
    STN_GPi = STN_GPi.tolist()
    # 6
    StrD1_SNc = np.array([dendrite.w for dendrite in StrD1SNc])
    StrD1_SNc[StrD1_SNc < 0] = 0.0
    StrD1_SNc = np.reshape(StrD1_SNc, (params["dim_STR"], 1))
    StrD1_SNc = StrD1_SNc.tolist()
    # 7
    IT_Thal = np.array([dendrite.w for dendrite in ITThal])
    IT_Thal[IT_Thal < 0] = 0.0
    if shortcut == False:
        IT_Thal = np.reshape(IT_Thal, (params["dim_IT"], 1))
    if shortcut == True:
        IT_Thal = np.reshape(IT_Thal, (params["dim_STR"], 1))
    IT_Thal = IT_Thal.tolist()
    # 8
    Thal_PFC = np.array([dendrite.w for dendrite in ThalPFC])
    Thal_PFC[Thal_PFC < 0] = 0.0
    Thal_PFC = np.reshape(Thal_PFC, (params["dim_Thal"], 1))
    Thal_PFC = Thal_PFC.tolist()
    # 9 Dopamine level
    DA = SNc.r
    DA = DA.tolist()

    gewichte = {
        "IT_StrD1": IT_StrD1,
        "IT_StrD2": IT_StrD2,
        "IT_STN": IT_STN,
        "StrD1_GPi": StrD1_GPi,
        "StrD2_GPe": StrD2_GPe,
        "STN_GPi": STN_GPi,
        "StrD1_SNc": StrD1_SNc,
        "IT_Thal": IT_Thal,
        "Thal_PFC": Thal_PFC,
        "DA": DA,
    }
    return gewichte


########################### record threshold ##################################


def extract_mean(ITStrD1, ITStrD2, ITSTN, StrD1GPi, StrD2GPe, STNGPi):
    mean_on = params["Mittelwert_umschalten"]

    # Mittelwerte aufzeichnen
    if mean_on:
        IT_StrD1 = np.mean(ITStrD1.pre.r) + params["ITStrD1_threshold_pre"]
        IT_StrD2 = np.mean(ITStrD2.pre.r) + params["ITStrD2_threshold_pre"]
        IT_STN = np.mean(ITSTN.pre.r) + params["ITSTN_threshold_pre"]
    else:
        IT_StrD1 = params["ITStrD1_threshold_pre"]
        IT_StrD2 = params["ITStrD2_threshold_pre"]
        IT_STN = params["ITSTN_threshold_pre"]

    StrD1_GPi = np.mean(StrD1GPi.post.r) + params["StrD1GPi_threshold_post"]
    StrD2_GPe = np.mean(StrD2GPe.post.r) + params["StrD2GPe_threshold_post"]
    STN_GPi = np.mean(STNGPi.post.r) + params["STNGPi_threshold_post"]

    mittelwerte = {
        "IT": [IT_StrD1, IT_StrD2, IT_STN],
        "GPi": [StrD1_GPi, STN_GPi],
        "GPe": [StrD2_GPe],
    }

    return mittelwerte


########################### read pre/post weigths #############################


def extract_mrx(ITStrD1, ITStrD2, ITSTN):
    IT_StrD1 = [ITStrD1.pre.r, ITStrD1.post.r]
    IT_StrD2 = [ITStrD2.pre.r, ITStrD2.post.r]
    IT_STN = [ITSTN.pre.r, ITSTN.post.r]

    for i in range(3):
        IT_StrD1[0] = np.append(IT_StrD1[0], ITStrD1.pre.r)
        IT_StrD2[0] = np.append(IT_StrD2[0], ITStrD2.pre.r)
        IT_STN[0] = np.append(IT_STN[0], ITSTN.pre.r)

    # mrx = [IT_StrD1, IT_StrD2, IT_STN]
    mrx = {"IT_StrD1": IT_StrD1, "IT_StrD2": IT_StrD2, "IT_STN": IT_STN}

    return mrx
