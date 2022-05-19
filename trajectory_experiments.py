# General
import numpy as np
import pickle
import tensorflow as tf
# GPFlow
import gpflow as gpf
from gpflow.utilities import print_summary, to_default_float
# Meta Model
from src.TrajectoryModel import TrajectoryModel
from src.TrajectoryModelFull import TrajectoryModelFull
from src.tools.utils import get_data_stats
from constants import HYPER_PRIORS_SCALED, HYPER_PRIORS_TOY, KERNEL_POOL, TOY_INCLUSION_PROB
# Kernel Selection
from src.experiments.base_kernels import initialize_base_kernels, polynomial_kernel_expansion
from src.KernelSelection import MultinomialKernels
from src.KernelSelection import log_marginal_likelihood as kernel_likelihood
from src.KernelSelection import approx_lml
from src.heartsteps.HSGP import HSGP
from src.tools.kernels import Interaction, kernel_to_string
from os import listdir
from os.path import isfile, join
# Globals
DIR = '/home/eura/Dropbox (Harvard University)/HeartStepsV1/HeartSteps/Eura/'

def run_experiment(traj, outdir, 
        n_gibbs_iters = 0, 
        n_seating = 0,
        mh_params = None, save_every = 1, start_i = 0):
    
    if start_i == 0: 
        traj.seat_customers()
        myfile = open(outdir + "iter-{}-model.pickle".format(start_i), "wb")
        pickle.dump(traj, myfile) # save initial model
        myfile.close()

    lmls = []
    trajectories = []
    for i in range(start_i + 1, start_i + n_gibbs_iters):
        print("Iteration ", i)
        traj.iterate(n_seating = n_seating, mh_params = mh_params)
        if i%save_every == 0:
            myfile = open(outdir + "iter-{}-model.pickle".format(i), "wb")
            pickle.dump(traj,myfile )
            myfile.close()
        
        # save lml
        lml = traj.posterior_likelihood()
        print("RECORDED LML {}\n".format(lml))
        lmls.append(lml)

        # save trajectory
        trajectory = [[traj.K[(m, t)] for t in range(traj.T[m])] for m in range(traj.M)]
        print(trajectory)
        trajectories.append(trajectory)

    res = {'trajectories': trajectories, 'lmls': lmls}
    return res

def resume_sampling(model_dir = "", n_gibbs_iters = 500, n_mh_iterations = 100, n_seating = 10, mh_hyper_proposal_variance = 0.1, burnin = 0.4, train_seed = 0):
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    # Find where we left off
    only_files = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) if ("iter" in f and "all" not in f)]
    file_iters = [int(f.split("-")[1]) for f in only_files if ("iter" in f and "all" not in f)]
    start_i = max(file_iters)
    n_remaining_iters = n_gibbs_iters - start_i
    print("REMAINING ITERATIONS {} FROM {}".format(n_remaining_iters, start_i)) 
    # Read in the model 
    model_file = "{}iter-{}-model.pickle".format(model_dir, start_i)
    print("Reading model from {}...".format(model_file))
    model = pickle.load(open(model_file, "rb"))
    
    # Continue sampling
    mh_params = {'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance, 'burnin':burnin}
    run_experiment(model, model_dir, start_i = start_i, mh_params = mh_params, n_gibbs_iters = n_remaining_iters, n_seating = n_seating)

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
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    # Initialize model seating and so on
    M, T = get_data_stats(X_list) 
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 

    # Initialize Base Distribution
    base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY) # base kernels
    poly_kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
    print([kernel_to_string(kernel) for kernel in poly_kernels])
    #p = np.array([component_inclusion_probability for k in poly_kernels])
    p = TOY_INCLUSION_PROB
    print(p)

    # TODO mean function, set not trainable!!!!
    # Initialize hyperparameter priors
    traj = TrajectoryModelFull(X_list, 
            y_list,  
            z_init = z_init,
            seed = train_seed,
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':False, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':poly_kernels, 'n_dimensions':X_list[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = HYPER_PRIORS_TOY,
            parent_kernel_prob = parent_kernel_prob, 
            adapt_noise_prior = adapt_noise_prior
            )
    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}

    # Run experiment
    res = run_experiment(traj, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))

def kernel_selection_toy(
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
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    # Initialize model seating and so on
    M, T = get_data_stats(X_list) 
    z_init = np.array([0 for m in range(np.sum(T))])
    res = []
    for m in range(M): 
        for t in range(T[m]): 
            res.append((m, t))
    reservations = {'': res} 

    # Initialize Base Distribution
    base_kernels = initialize_base_kernels(X_list[(0,0)], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY) # base kernels
    poly_kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
    print([kernel_to_string(kernel) for kernel in poly_kernels])
    #p = np.array([component_inclusion_probability for k in poly_kernels])
    p = TOY_INCLUSION_PROB
    print(p)

    # TODO mean function, set not trainable!!!!
    # Initialize hyperparameter priors
    traj = TrajectoryModel(X_list, 
            y_list,  
            z_init = z_init,
            seed = train_seed,
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':False, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':poly_kernels, 'n_dimensions':X_list[(0,0)].shape[1]},
            alpha = alpha, 
            hyper_priors = HYPER_PRIORS_TOY,
            parent_kernel_prob = parent_kernel_prob, 
            adapt_noise_prior = adapt_noise_prior
            )
    mh_params = {'burnin':mh_burnin, 'num_iterations':n_mh_iterations, 'hyper_proposal_variance':mh_hyper_proposal_variance}

    # Run experiment
    res = run_experiment(traj, outdir, n_gibbs_iters = n_gibbs_iters, n_seating = n_seating, mh_params = mh_params)
    pickle.dump(res, open(outdir + "res.pickle", "wb"))

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

