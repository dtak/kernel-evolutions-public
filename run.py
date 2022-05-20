import json
from pathlib import Path
import os
import sys
import csv
import pickle
import itertools
#import trajectory_experiments
#import pure_selection_experiments
#import multitask_selection_experiments
from experiment_settings.toy import TOY_PARAMS

import optparse
import numpy as np
import tensorflow as tf

from src.tools.base_kernels import initialize_base_kernels, polynomial_kernel_expansion
from src.tools.experiments import get_real_data, get_toy_data, get_path_from_setting, get_user_data, get_data_stats
from constants import KERNEL_POOL, HYPER_PRIORS_TOY, HYPER_PRIORS_SCALED, TOY_INCLUSION_PROB


import models

def safe_zip(*args):
    if len(args) > 0:
        first = args[0]
        for a in args[1:]:
            assert(len(a) == len(first))

    return list(zip(*args))

def run_experiment(method, data, outdir, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    2. Get your experiment's parameters from 'exp_kwargs'
    3. Run your experiment
    4. Store the results however you see fit in 'exp_dir'
    '''
    print('Running experiment with method {}, data {} output to {}...'.format(method, data, outdir))
    
    try: 
        seed = exp_kwargs["train_seed"]
    except: 
        print("Please make sure to define all necessary experiment parameters")
 
    
    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
   
    # Get data set, base distribution args, hyperpriors
    if data == "toy": 
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(data, exp_kwargs) # generate synthetic data
        hyper_priors = HYPER_PRIORS_TOY
       
        # base distribution args
        base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = hyper_priors)
        kernel_components = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2)
        p = TOY_INCLUSION_PROB

    else: 
        if data == "air" or data == "pollutants":
            X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir =  "./data/uci/{}/".format(data), scaleall = True, shuffle = False) # ordering matters
        else: 
            X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/{}/".format(data), scaleall = True, shuffle = True) # no time
        
        hyper_priors = HYPER_PRIORS_SCALED

        M, T = get_data_stats(X_list) 
        X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])
        kernel_components = KERNEL_POOL[dataset](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors, interaction = exp_kwargs['interaction'])
    n_dimensions = X_list[(0,0)].shape[1]

    # Make output path for this experiment
    outdir = get_path_from_setting(method, data, exp_kwargs, users, outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True) 

    # Create the model
    if method == "evolution": 
        # Get evolution specific parameters
        try: 
            alpha = exp_kwargs['alpha']
            parent_kernel_prob = exp_kwargs['parent_kernel_prob']
            adapt_noise_prior = exp_kwargs['adapt_noise_prior']

        except: 
            print("Please specify all evolution relevant parameters")

        
        # Create model
        model = models.create_evolution_model(X_list, y_list,
            seed = seed, # Defined above
            p = p, 
            kernel_components = kernel_components, 
            n_dimensions = n_dimensions,
            hyper_priors = hyper_priors,
            alpha = alpha, # Evolution specific
            parent_kernel_prob = parent_kernel_prob, 
            adapt_noise_prior = adapt_noise_prior)
    
    elif method == "stratified":
        # Get specific parameters
        try: 
            alpha = exp_kwargs['alpha']
        except: 
            print("Please specify all final relevant parameters")
    
        # Create model
        model = models.create_stratified_model(X_list, y_list, 
            seed = seed, # Defined above
            p = p, 
            kernel_components = kernel_components, 
            n_dimensions = n_dimensions,
            hyper_priors = hyper_priors,
            alpha = alpha)

    elif method == "final": 
        # Get final specific parameters
        try: 
            alpha = exp_kwargs['alpha']
        except: 
            print("Please specify all final relevant parameters")
        
        # Create model
        model = models.create_final_model(X_list, y_list, 
            seed = seed,
            p = p, 
            kernel_components = kernel_components, 
            n_dimensions = n_dimensions,
            hyper_priors = hyper_priors,
            alpha = alpha) 
    elif method == "memoryless":
        try: 
            opt_params = {
                    'n_mh_iterations': exp_kwargs['n_mh_iterations'],
                    'mh_burnin': exp_kwargs['mh_burnin'],
                    'hyper_proposal_variance': exp_kwargs['hyper_proposal_variance']

                    }
        except: 
            print("Please specify memoryless optimization parameters")
        model = models.create_memoryless_model(X_list, y_list, 
                kernel_components = kernel_components,
                p = p, 
                hyper_priors = hyper_priors,
                n_dimensions = n_dimensions,
                seed = seed)

    else: # method is ARD
        model = models.create_ard_model(X_list, y_list, hyper_priors = hyper_priors)
        try: 
            opt_params = {
                    'n_restarts': exp_kwargs['n_restarts'],
                    'hyper_proposal_variance': exp_kwargs['hyper_proposal_variance']
                    }
        except: 
            print("Please specify ARD optimization parameters")

    # Run experiment
    print("Running experiment...")
    if method == "evolution" or method == "stratified" or method == "final":
        # Get meta-model parameters
        try: 
            mh_burnin = exp_kwargs['mh_burnin']
            n_mh_iterations = exp_kwargs['n_mh_iterations']
            mh_hyper_proposal_variance = exp_kwargs['mh_hyper_proposal_variance']
            n_gibbs_iters = exp_kwargs['n_gibbs_iters']
            n_seating = exp_kwargs['n_seating']
        except: 
            print("Please specify all meta-model relevant parameters")
        mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}
        
        # Meta-training 
        models.train_meta_model(model, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    elif method == "ard" or method == "memoryless": 
        try: 
            user  = exp_kwargs['user']
        except: 
            print("Make sure to specify a user for the single task methods.")
        models.predict_test_user(model, X_list, y_list, user,  opt_params, outdir)


if __name__ == '__main__':
    p = optparse.OptionParser()
    p.add_option('--method', '-m')
    p.add_option('--data', '-d')
    
    # Get args
    (opt, args) = p.parse_args()
    if opt.method is None or opt.data is None: 
        print("Correct usage is python submit_batch.py -n <method> -d <data>")
    data = opt.data 
    method = opt.method

    # Get output dir
    outdir = "results/{}/{}/".format(method, data)
    
    # Get experiment settings 
    try:
        if data == "toy": 
            params = TOY_PARAMS[method]
    except: 
            print("Please specify a valid experiment.")

    # For each experiment, create a job for every combination of parameters
    for vals in itertools.product(*list(params.values())):
        exp_kwargs = dict(safe_zip(params.keys(), vals))
        run_experiment(method, data, outdir, exp_kwargs)
