# General
import numpy as np
import pickle
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import time 

# GPFlow
import gpflow as gpf
from gpflow.utilities import print_summary, to_default_float

# Import baseline object classes
from src.TrajectoryModel import TrajectoryModel
from src.StratifiedModel import StratifiedModel
from src.ARD import ARD
from src.Memoryless import Memoryless
from src.tools.utils import get_data_stats
from src.tools.experiments import get_data_stats

# Kernel Selection
from src.KernelSelection import MultinomialKernels
from src.KernelSelection import log_marginal_likelihood as kernel_likelihood
from src.KernelSelection import approx_lml
from src.tools.kernels import Interaction, kernel_to_string

def train_meta_model(traj, outdir, 
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

def predict_test_user(model, X_list, y_list, user, opt_params, outdir): 
    M, T = get_data_stats(X_list)
    for t in range(T[user]):
        print("Predicting at timestep", t)
        # start at model from previous time step
        if t > 0:
            model.update_data(X_list[(user, t)], y_list[(user, t)])
        
        # Optimize
        start_time = time.time()
        model.optimize(**opt_params)
        end_time = time.time()
        runtime = end_time - start_time
        
        # Pickle files
        results = {"model":model, "runtime":runtime}
        with open("{}chunk_{}.pickle".format(outdir, t), 'wb') as handle:
            pickle.dump(results, handle)
def create_stratified_model(X_list, y_list, 
            seed = 0, 
            p = None, 
            kernel_components = None, 
            n_dimensions = 0,
            hyper_priors = None,
            alpha = 1): 
    model = StratifiedModel(X_list,
            y_list, 
            seed = seed, 
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':False, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernel_components, 'n_dimensions':n_dimensions},
            alpha = alpha, 
            hyper_priors = hyper_priors)
    return model

def create_final_model(X_list, y_list, 
            seed = 0,
            p = None, 
            kernel_components = None, 
            n_dimensions = 0,
            hyper_priors = None,
            alpha = 1): 
    
    # gets the data set at the last time step 
    M, T = get_data_stats(X_list)
    X_list_total = {(m, 0): X_list[(m, T[m] - 1)] for m in range(M)}
    y_list_total = {(m, 0): y_list[(m, T[m] - 1)] for m in range(M)}
    
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
            seed = seed, 
            reservations = reservations,
            model_to_string = kernel_to_string, 
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':False, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            base_distribution_args = {'p':p, 'components':kernel_components, 'n_dimensions':n_dimensions},
            alpha = alpha, 
            hyper_priors = hyper_priors, 
            parent_kernel_prob = None)
    
    return model

def create_memoryless_model(X_list, y_list,
        kernel_components = None,
        p = 0.1, 
        hyper_priors = None, 
        n_dimensions = None,
        seed = 0):
    base_dist =  MultinomialKernels(components = kernel_components, p = p, hyper_priors = hyper_priors, n_dimensions = n_dimensions)
    model = Memoryless(X_list[(0,0)], y_list[(0,0)], # data will get updated incrementally as observed
            base_dist, likelihood = kernel_likelihood, 
            likelihood_params = {'heartsteps': False, 'mean_function':gpf.mean_functions.Zero()}, seed = seed)
    return model
       

def create_evolution_model(X_list, y_list, 
            seed = 0, # Defined above
            reservations = None,
            p = None, 
            kernel_components = None, 
            n_dimensions = 0,
            hyper_priors = None,
            alpha = 1, # Evolution specific
            parent_kernel_prob = 1, 
            adapt_noise_prior = True): 
        
         # Initialize customer assignments 
        M, T = get_data_stats(X_list) 
        z_init = np.array([0 for m in range(np.sum(T))])
        res = []
        for m in range(M): 
            for t in range(T[m]): 
                res.append((m, t))
        reservations = {'': res} 

        model = TrajectoryModel(X_list, 
            y_list,  
            seed = seed, # Defined above
            z_init = z_init,
            reservations = reservations,
            base_distribution_args = {'p':p, 'components':kernel_components, 'n_dimensions':n_dimensions},
            hyper_priors = hyper_priors,
            model_to_string = kernel_to_string, # Fixed for most
            likelihood_func = kernel_likelihood,
            likelihood_params = {'heartsteps':False, 'mean_function':gpf.mean_functions.Zero()},
            base_distribution_constructor = MultinomialKernels, 
            alpha = alpha, # Evolution specific
            parent_kernel_prob = parent_kernel_prob, 
            adapt_noise_prior = adapt_noise_prior)
        return model

def create_ard_model(X_list, y_list, hyper_priors = None): 
    return ARD(X_list[(0,0)], y_list[(0,0)], hyper_priors = hyper_priors)
