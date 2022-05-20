# General
import numpy as np
import pickle
import tensorflow as tf
from os import listdir
from os.path import isfile, join

# GPFlow
import gpflow as gpf
from gpflow.utilities import print_summary, to_default_float

# Import baseline object classes
from src.TrajectoryModel import TrajectoryModel
from src.StratifiedModel import StratifiedModel

from src.tools.utils import get_data_stats

# Kernel Selection
#from src.experiments.base_kernels import initialize_base_kernels, polynomial_kernel_expansion
from src.KernelSelection import MultinomialKernels
from src.KernelSelection import log_marginal_likelihood as kernel_likelihood
from src.KernelSelection import approx_lml
#from src.heartsteps.HSGP import HSGP
from src.tools.kernels import Interaction, kernel_to_string

# Globals
from constants import HYPER_PRIORS_SCALED, HYPER_PRIORS_TOY, KERNEL_POOL, TOY_INCLUSION_PROB



def kernel_selection_toy_full(
        X_list, y_list, 
        n_gibbs_iters = 10,
        n_seating = 5 ,
        n_mh_iterations = 50, 
        mh_burnin = 0.4, 
        mh_hyper_proposal_variance = 0.1, 
        alpha = 1,
        adapt_noise_prior = False,
        component_inclusion_probability = 0.1,
        parent_kernel_prob = 1, 
        train_seed = 0, 
        outdir = "/home/eura/Dropbox (Harvard University)/HeartStepsV1/HeartSteps/Eura/icml/kernel-selection/toy/"
        ):

    # Initialize model seating and so on
    M, T = get_data_stats(X_list) 
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 

    # Initialize Base Distribution
    print([kernel_to_string(kernel) for kernel in poly_kernels])
    #p = np.array([component_inclusion_probability for k in poly_kernels])
    p = TOY_INCLUSION_PROB
    print(p)



# TODO: resample the set of training users
def kernel_selection_real(
        X_list, y_list,
        n_gibbs_iters = 50,
        n_seating = 5 ,
        n_mh_iterations = 50, 
        mh_burnin = 0.1, 
        mh_hyper_proposal_variance = 0.1, 
        component_inclusion_probability = 0.1,
        adapt_noise_prior = False,
        parent_kernel_prob = 1, 
        outdir = "/home/eura/Dropbox (Harvard University)/HeartStepsV1/HeartSteps/Eura/icml/kernel-selection/hs/", 
        alpha = 5, 
        train_seed = 0,
        interaction = True,
        dataset = "heartsteps"
        ):
    
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    # Initialize model seating and so on
    M, T = get_data_stats(X_list) 
    print("M: {}\nT: {}".format(M, T))
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 
    print("Initial z: ", z_init)

    # Initialize Base Distribution
    X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])
    heartsteps = dataset == "heartsteps" # see if heartsteps
    if dataset == "toy" or dataset == "multi": 
        base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY)
        kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
        hyper_priors = HYPER_PRIORS_TOY
        print("BASE SHAPE: ", len(kernels), [kernel_to_string(k) for k in kernels])
    else: # real data
        hyper_priors = HYPER_PRIORS_SCALED
        kernels = KERNEL_POOL[dataset](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors,  interaction = interaction)

    p = np.ones(len(kernels)) * component_inclusion_probability
    kernel_product = np.array([isinstance(kernel, gpf.kernels.Product) for kernel in kernels])
    p[kernel_product] = component_inclusion_probability/5.
    print("{} TOTAL CANDIDATE KERNELS".format(len(kernels)), p)
    # TODO mean function, set not trainable!!!!
    # Initialize model
    traj = TrajectoryModel(X_list, 
            y_list,  
            z_init = z_init,
            seed = train_seed, 
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':heartsteps, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernels, 'n_dimensions':X_list[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = hyper_priors,
            adapt_noise_prior = adapt_noise_prior,
            parent_kernel_prob = parent_kernel_prob
            )
    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}

    res = run_experiment(traj, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))


def kernel_selection_real_full(
        X_list, y_list,
        n_gibbs_iters = 50,
        n_seating = 5 ,
        n_mh_iterations = 50, 
        mh_burnin = 0.1, 
        mh_hyper_proposal_variance = 0.1, 
        component_inclusion_probability = 0.1,
        adapt_noise_prior = False,
        parent_kernel_prob = 1, 
        outdir = "/home/eura/Dropbox (Harvard University)/HeartStepsV1/HeartSteps/Eura/icml/kernel-selection/hs/", 
        alpha = 5, 
        train_seed = 0,
        interaction = True,
        dataset = "heartsteps"
        ):
    
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    # Initialize model seating and so on
    M, T = get_data_stats(X_list) 
    print("M: {}\nT: {}".format(M, T))
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 
    print("Initial z: ", z_init)

    # Initialize Base Distribution
    X_all = np.vstack([X_list[(m, T[m] - 1)] for m in range(M)])
    heartsteps = dataset == "heartsteps" # see if heartsteps
    if dataset == "toy" or dataset == "multi": 
        base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY)
        kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
        hyper_priors = HYPER_PRIORS_TOY
        print("BASE SHAPE: ", len(kernels), [kernel_to_string(k) for k in kernels])
    else: # real data
        hyper_priors = HYPER_PRIORS_SCALED
        kernels = KERNEL_POOL[dataset](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors,  interaction = interaction)

    p = np.ones(len(kernels)) * component_inclusion_probability
    kernel_product = np.array([isinstance(kernel, gpf.kernels.Product) for kernel in kernels])
    p[kernel_product] = component_inclusion_probability/5.
    print("{} TOTAL CANDIDATE KERNELS".format(len(kernels)), p)
    # TODO mean function, set not trainable!!!!
    # Initialize model
    traj = TrajectoryModelFull(X_list, 
            y_list,  
            z_init = z_init,
            seed = train_seed, 
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':heartsteps, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernels, 'n_dimensions':X_list[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = hyper_priors,
            adapt_noise_prior = adapt_noise_prior,
            parent_kernel_prob = parent_kernel_prob
            )
    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}

    res = run_experiment(traj, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))

