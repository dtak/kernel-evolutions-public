from src.tools.utils import get_data_stats
from trajectory_experiments import run_experiment
from src.KernelSelection import MultinomialKernels
from src.KernelSelection import log_marginal_likelihood as kernel_likelihood
from src.tools.kernels import kernel_to_string
from src.ClusterSelection import ClusterSelection
from src.StratifiedModel import StratifiedModel
from src.TrajectoryModel import TrajectoryModel
from src.experiments.base_kernels import initialize_base_kernels, polynomial_kernel_expansion
from constants import HYPER_PRIORS_SCALED, HYPER_PRIORS_TOY,KERNEL_POOL,TOY_INCLUSION_PROB 
from src.heartsteps.HSGP import HSGP

import numpy as np
import tensorflow as tf
import gpflow as gpf
import pickle
def stratified_selection(X_list, y_list,
        simple_kernel_cutoff = 3,
        n_gibbs_iters = 10,
        n_seating = 5 ,
        n_mh_iterations = 50, 
        mh_burnin = 0.4, 
        adapt_noise_prior = False,
        mh_hyper_proposal_variance = 0.1, 
        alpha = 1,
        component_inclusion_probability = 0.1,
        train_seed = 0, 
        interaction = True,
        outdir= "", 
        data = "" ): 
    '''
    for each time step, cluster, find model
    '''
    # seed
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)
    
    # Clustering params
    level_options = {}
    n_features = X_list[(0,0)].shape[1]
    M, T = get_data_stats(X_list)
    X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])

    heartsteps = data == "heartsteps" # see if heartsteps
    if data == "toy" or data == "multi": 
        base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY)
        kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
        hyper_priors = HYPER_PRIORS_TOY
        p = TOY_INCLUSION_PROB
        print("BASE SHAPE: ", len(kernels), [kernel_to_string(k) for k in kernels])
    else: # real data
        hyper_priors = HYPER_PRIORS_SCALED
        kernels = KERNEL_POOL[data](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors,  interaction = interaction)
        p = np.ones(len(kernels)) * component_inclusion_probability 
        kernel_product = np.array([isinstance(kernel, gpf.kernels.Product) for kernel in kernels])
        p[kernel_product] = component_inclusion_probability/5.
    
    print(p)
    model = StratifiedModel(X_list, 
            y_list,  
            seed = train_seed, 
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':heartsteps, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernels, 'n_dimensions':X_list[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = hyper_priors)

    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}
    
    # Run experiment
    res = run_experiment(model, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))

def cluster_selection(X_list, y_list, 
        n_gibbs_iters = 10,
        n_seating = 5 ,
        n_mh_iterations = 50, 
        mh_burnin = 0.4, 
        mh_hyper_proposal_variance = 0.1, 
        alpha = 1,
        adapt_noise_prior = False,
        component_inclusion_probability = 0.1,
        train_seed = 0, 
        interaction = True,
        outdir= "", 
        data = "" ): 
    '''
    for full data sets, cluster, find model
    '''
    # seed
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)
    
    # gets the data set at the last time step 
    M, T = get_data_stats(X_list)
    X_list_total = {(m, 0): X_list[(m, T[m] - 1)] for m in range(M)}
    y_list_total = {(m, 0): y_list[(m, T[m] - 1)] for m in range(M)}
#    X_list_total = {(m, 0): X_list[(m, 0)] for m in range(M)}
#    y_list_total = {(m, 0): y_list[(m, 0)] for m in range(M)}
    print(X_list_total[(0,0)].shape)
    X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])

    # initialize kernels
    X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])
    heartsteps = data == "heartsteps" # see if heartsteps
    if data == "toy" or data == "multi": 
        base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY)
        kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
        hyper_priors = HYPER_PRIORS_TOY
        p = TOY_INCLUSION_PROB
        print("BASE SHAPE: ", len(kernels), [kernel_to_string(k) for k in kernels])
    else: # real data
        hyper_priors = HYPER_PRIORS_SCALED
        kernels = KERNEL_POOL[data](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors,  interaction = interaction)
        p = np.ones(len(kernels)) * component_inclusion_probability
        kernel_product = np.array([isinstance(kernel, gpf.kernels.Product) for kernel in kernels])
        p[kernel_product] = component_inclusion_probability/5.

    
    # Initialize model seating and so on
    print(p)
    M, T = get_data_stats(X_list_total) 
    print("M: {}\nT: {}".format(M, T))
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 

    model = TrajectoryModel(X_list_total, 
            y_list_total,  
            z_init = z_init,
            seed = train_seed, 
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':heartsteps, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernels, 'n_dimensions':X_list_total[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = hyper_priors, 
            parent_kernel_prob = None
            )

    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}
    
    # Run experiment
    res = run_experiment(model, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))
