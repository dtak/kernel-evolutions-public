import os
import copy
import sys
import itertools
import json
import subprocess
import socket
import optparse

from specify_experiment import run_experiment    

#################################
##       CONSTANTS             ##
#################################
RESULT_DIR = "./results/"
from experiment_settings.toy import TOY_PARAMS



TMP_DIR = 'tmp'
TEMPLATE = 'template.sh'
DRYRUN = True

def safe_zip(*args):
    if len(args) > 0:
        first = args[0]
        for a in args[1:]:
            assert(len(a) == len(first))

    return list(zip(*args))



def main(EXP_TYPE = "", METHOD = "", DATA = ""):
    # Create the directory where your results will go.
    # In this directory you can make a sub-directory for each experiment you run.
    # Experiments are listed in the 'QUEUE' variable above
    QUEUE = []
        QUEUE.append(
            )
    #####################################
    ######## TOY KERNEL SELECTION########
    #####################################
    if DATA == "toy" and EXP_TYPE == "kernel-selection": 
        try: 
            experiment_settings = TOY_PARAMS[METHOD] 
            output_dir = RESULT_DIR + "{}/{}".format(DATA, METHOD)
        except e: 
            print("Please run a valid toy experiment.")
    elif DATA == ""
    
    method_name = "{}_kernel_selection_{}".format(METHOD, DATA)

    ####################################
    ####### POLLUTANTS          ########
    ####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/cluster"
        method_name = "cluster_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/traj-full"
        method_name = "traj_kernel_selection_pollutants_full"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                interaction = [True],
                adapt_noise_prior = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/traj"
        method_name = "traj_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                interaction = [True],
                adapt_noise_prior = [True],
                train_seed = [1], 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = [0]
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/stratified"
        method_name = "stratified_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "pollutants": 
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/ard"
        method_name = "ard_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [24], # Data args 
                chunk_size = [5],
                user = range(24),
                data_seed = [0]
                )
            ))
    ####################################
    ####### AIR                 ########
    ####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "air":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/air/cluster"
        method_name = "cluster_kernel_selection_air"
        QUEUE.append(
            ('uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "air":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/air/traj"
        method_name = "traj_kernel_selection_air"
        QUEUE.append(
            ('uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "air":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/air/traj-full"
        method_name = "traj_kernel_selection_air_full"
        QUEUE.append(
            ('uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "air":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/air/stratified"
        method_name = "stratified_kernel_selection_air"
        QUEUE.append(
            ('uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "air": 
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/air/ard"
        method_name = "ard_kernel_selection_air"
        QUEUE.append(
            ('uci', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [12], # Data args 
                chunk_size = [5],
                user = range(12),
                data_seed = [0]
                )
            ))
    ####################################
    ####### CHLORIDES           ########
    ####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/cluster"
        method_name = "cluster_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [21], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/traj-full"
        method_name = "traj_kernel_selection_chlorides_full"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/traj"
        method_name = "traj_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/stratified"
        method_name = "stratified_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "chlorides": 
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/ard"
        method_name = "ard_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [31], # Data args 
                chunk_size = [5],
                user = range(31),
                data_seed = range(3)
                )
            ))
    #####################################
    ######## WINE                ########
    #####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/traj-full"
        method_name = "traj_kernel_selection_wine_full"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                adapt_noise_prior = [True],
                interaction = [True],
                parent_kernel_prob = [0.9], 
                train_seed = range(3), 
                M = [21], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/traj"
        method_name = "traj_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                adapt_noise_prior = [True],
                interaction = [True],
                parent_kernel_prob = [0.9], 
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/cluster"
        method_name = "cluster_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = [0], 
                M = [21], # Data args 
                chunk_size = [5],
                data_seed = [0,1]
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/stratified"
        method_name = "stratified_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "mh" and DATA == "wine":
        OUTPUT_DIR = HOME_DIR + "icml/kernel-selection/wine/mh"
        method_name = "mh_kernel_selection_wine"
        QUEUE.append(
            ('stricter', dict(
                n_iters = [10000], # Method args
                hyper_proposal_variance = [0.05],
                component_inclusion_probability = [0.05],
                train_seed = range(3),   
                M = [12], # Data args 
                chunk_size = [50],
                user = range(12),
                data_seed = [0]
            ))
        )
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "wine": 
        OUTPUT_DIR = HOME_DIR + "icml/kernel-selection/wine/ard"
        method_name = "ard_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [31], # Data args 
                chunk_size = [5],
                user = range(31),
                data_seed = range(3)
                )
            ))
    else: 
        raise ValueError("Please enter a valid experiment type.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create a temporary directory for the slurm scripts
    # that are automatically generated by this script
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    # Read in the template so we can modify it programmatically
    with open(TEMPLATE, 'r') as f:
        template = f.read()

    # For each experiment, create a job for every combination of parameters
    for experiment_name, params in QUEUE:
        for vals in itertools.product(*list(params.values())):
            exp_kwargs = dict(safe_zip(params.keys(), vals))
            run_experiment(method_name, OUTPUT_DIR, template, experiment_name, exp_kwargs)

if __name__ == '__main__':
    p = optparse.OptionParser()
    p.add_option('--experiment', '-e', default = "kernel-selection")
    p.add_option('--method', '-m')
    p.add_option('--data', '-d')
    
    (opt, args) = p.parse_args()
    print(opt, " ", args)
    if opt.method is None or opt.data is None: 
        print("Correct usage is python submit_batch.py -n <method> -d <data>")

    main(EXP_TYPE = opt.experiment, METHOD =opt.method, DATA = opt.data)
