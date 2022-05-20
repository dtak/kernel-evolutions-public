import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from gpflow.utilities import set_trainable, to_default_float
import gpflow as gpf
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from src.tools.utils import split_data_by_size, deepcopy, get_data_stats
from src.KernelSelection import log_marginal_likelihood
from src.tools.kernels import create_linear, create_period, create_rbf, Linear

def get_user_data(user_ind, X_list, y_list): 
    # Get data for this user at each time step
    M, T = get_data_stats(X_list)

    print("reading user_ind {} with {} timesteps.".format(user_ind, T[user_ind]))
    X_list_user = []
    y_list_user = []
    for t in range(T[user_ind]):
        X_list_user.append(X_list[(user_ind, t)])
        y_list_user.append(y_list[(user_ind, t)])
    return X_list_user, y_list_user

def args_to_dir(exp_kwargs): 
    string = ""
    i = 0
    for key, val in exp_kwargs.items(): 
        if i == 0: 
            string += "{}-{}".format(key, val)
        else: 
            string += "-{}-{}".format(key, val)
        i += 1

    return string + "/"


def get_path_from_setting(method, data, exp_kwargs, users, outdir): 
    is_singletask = method == "memoryless" or method == "ard"
    
    # Path differs depending on multitask or single task
    if is_singletask:
        # pure selection method runs only for specified user
        user_ind = exp_kwargs['user']
        if data == "toy": 
            user = user_ind
        else: 
            user = users[user_ind]
        copy_kwargs = exp_kwargs.copy() 
        del copy_kwargs['user']
        outdir += args_to_dir(copy_kwargs)
        outdir += "user_{}/".format(user) 
        Path("{}".format(outdir)).mkdir(parents=True, exist_ok=True)
    else:
        # Make sure output path exists
        outdir += args_to_dir(exp_kwargs)
        Path("{}".format(outdir)).mkdir(parents=True, exist_ok=True)
        # write users to file
        with open(outdir + "users.csv", "w") as f:
            for user in users:
                f.write(str(user) + "\n")

    return outdir


def get_real_data(method_name, exp_kwargs, data_dir = None, scaleall = True, shuffle = False, time_column = None): 
    seed = exp_kwargs['data_seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    
    # Read data
    X_all = pickle.load(open(data_dir + "X_all.pickle", "rb"))
    df = pickle.load(open(data_dir + "df.pickle", "rb"))
#    print("UNIQUE USERS ", df.user.unique().shape[0]) 
    if time_column is None: 
        M = exp_kwargs['M']
        chunk_size = exp_kwargs['chunk_size']
    
        # Sample users
        users = np.random.choice(df['user'].unique(), M, replace = False)
    else: 
        timestep_min = exp_kwargs['timestep_min']
        times = df[time_column].unique()
        times.sort()
        points_per_timestep = df.groupby(['user', time_column]).size().unstack()
        users = points_per_timestep.index[points_per_timestep.apply(lambda x: (~np.any(np.isnan(x))) & (np.min(x) >= timestep_min), axis = 1)]
        
    print("Using {}  users {}".format(users.shape[0], users))
    
    # Scaler defined by data from all training tasks
    if scaleall:  
        print("Scaling with all users, timesteps!")
        inds = []
        for m, user in enumerate(users): 
            # get data
            user_ind = df['user'] == user
            inds.extend(np.where(user_ind)[0])
        X_train_all = X_all[inds, :]
        print("Scaler min max: ")
        print(min(inds), "  ", max(inds))
        print("Data Shape: ", X_all.shape)
        scaler = MinMaxScaler().fit(X_train_all)
        X_scaled = scaler.transform(X_train_all)
    
    # Data from each task
    X_list = {}
    y_list = {}
    for m, user in enumerate(users): 
        # get data
        user_ind = df['user'] == user
        X_user = X_all[user_ind, :]
        y_user = df.loc[user_ind, 'y'].values.reshape(-1, 1)
        # Shuffle data 
        if shuffle:
            shuffle_ind = np.random.choice(range(X_user.shape[0]), X_user.shape[0], replace = False)
            X_user = X_user[shuffle_ind, :] 
            y_user = y_user[shuffle_ind, :]

        # time split
        if time_column is None: 
            # Split by chunks
            X_user_list, y_user_list = split_data_by_size(X_user, y_user, size = chunk_size)
        else:
            user_df = df.loc[user_ind, :]
            # Split by column which denotes time
            X_user_list = []
            y_user_list = []
            for time in times:
                time_indices = user_df[time_column] == time
                X_user_list.append(X_user[time_indices, :])
                y_user_list.append(y_user[time_indices, :])

        T = len(y_user_list)
        # for each timestep
        for t in range(T):
            # training is CUMULATIVE set
            X_train = np.vstack(X_user_list[:(t+1)])
            y_train = np.vstack(y_user_list[:(t+1)])
    #        print(X_train.shape)
            if not scaleall: # scaler defined by individual training task
                scaler = MinMaxScaler().fit(X_train)
            
            X_train = scaler.transform(X_train)

            if 'heartsteps' in method_name: 
                X_train[:, 4] = 1 # bias action
                X_train[:, 5] = 1 # bias

            X_list[(m, t)] = X_train 
            y_list[(m, t)] = y_train
    return X_list, y_list, users, exp_kwargs

def sample_periodic(mean_period, mean_amp, mean_lengthscale):
    period = np.random.lognormal(mean = np.log(mean_period), sigma = 0.1)
    amplitude = np.random.lognormal(mean = np.log(mean_amp), sigma = 0.1)
    lengthscale = np.random.lognormal(mean = np.log(mean_lengthscale), sigma = 0.1) 
    periodic_kernel = gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = lengthscale, variance = amplitude), period = period)
    return periodic_kernel

def sample_linear(mean_amp): 
    amplitude = np.random.lognormal(mean = np.log(mean_amp), sigma = 0.1)
    shift = np.random.normal(5, 1)
    linear_kernel = Linear(variance = amplitude, location = shift) 
    return linear_kernel

def sample_rbf(mean_lengthscale, mean_amp): 
    amplitude = np.random.lognormal(mean = np.log(mean_amp), sigma = 0.1)
    lengthscale = np.random.lognormal(mean = np.log(mean_lengthscale), sigma = 0.1) 
    rbf_kernel =  gpf.kernels.RBF(lengthscales = lengthscale, variance = amplitude)
    return rbf_kernel

def get_toy_data(method_name, exp_kwargs): 
    '''
    DATA ARGS: 
    random seed:
    noise: 
    n_users: 
    num_per_timestep: 
    '''
    M = exp_kwargs['M']
    ground_sigma = exp_kwargs['noise']
    seed = exp_kwargs['data_seed'] 
    num_per_timestep = exp_kwargs['num_per_timestep']
    ground_kernel = exp_kwargs['ground_kernel']

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate data
    X_list = {}
    y_list = {}
    y_test_list = []
    X_test = np.linspace(0, 20, 200).reshape(-1, 1)
    T = len(num_per_timestep)
    zero_centered = to_default_float((np.zeros((1, 1)), np.zeros((1, 1))))
    f_list = {}

    # Show data
    total_lml = 0
    for m in range(M): 
        if ground_kernel == "similar": # Similar, synthetic experiment 
            if m < M//2: # LIN + PER
                periodic_kernel = sample_periodic(4, 5, 1)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
            else: # LIN + SE
                rbf_kernel = sample_rbf(3, 5)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + rbf_kernel)
        elif ground_kernel == "complex": # Complex, synthetic experiment
            if m < M//2: # LIN * PER
                periodic_kernel = sample_periodic(5, 1, 1)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel * periodic_kernel)
            else: # SE * PER
                periodic_kernel = sample_periodic(3, 5, 1)
                rbf_kernel = sample_rbf(5, 4)
                rbf_kernel2 = sample_rbf(10, 7)
                gp = gpf.models.GPR(zero_centered, periodic_kernel * rbf_kernel + rbf_kernel2)
        elif ground_kernel == "distinct": # Distinct, synthetic experiment
            if m < M//2:
                period = np.random.lognormal(mean = np.log(4), sigma = 0.1)
                amplitude = np.random.lognormal(mean = np.log(5), sigma = 0.1)
                lengthscale = np.random.lognormal(mean = 0, sigma = 0.1) 
                periodic_kernel = gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = lengthscale, variance = amplitude), period = period)

                amplitude = np.random.lognormal(mean = np.log(4), sigma = 0.1)
                shift = np.random.normal(5, 1)
                linear_kernel = Linear(variance = amplitude, location = shift) 
                
                gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
            else: 
                amplitude = np.random.lognormal(mean = np.log(10), sigma = 0.1)
                lengthscale = np.random.lognormal(mean = 0, sigma = 0.1) 
                gp = gpf.models.GPR(zero_centered, gpf.kernels.RBF(lengthscales = lengthscale, variance = amplitude))
        
        gp.likelihood.variance.assign(ground_sigma)
        #gpf.utilities.print_summary(gp)
        
        # sample a function from the GP
        X_train = np.random.uniform(0, 20, np.sum(num_per_timestep)).reshape(-1, 1)
        X = np.vstack((X_train, X_test))
        f = gp.predict_f_samples(to_default_float(X)) 
        y = f + np.random.normal(0, ground_sigma, size = f.shape)
        y_test_list.append(y[(-1 * X_test.shape[0]):, :])
        f_list[m] = f

        # timesteps
        for t in range(T): 
            num_points = np.sum(num_per_timestep[:(t + 1)])
            X_total = X[:num_points, :]
            y_total = y[:num_points, :]

            X_list[(m, t)] = X_total
            y_list[(m, t)] = y_total
        
            total_lml += log_marginal_likelihood(X[:num_points, :], y[:num_points, :], {'structure': gp.kernel, 'noise': ground_sigma})

            upto = int(np.sum(num_per_timestep[:t]))
    if 'num_per_timestep' in exp_kwargs: 
        del exp_kwargs['num_per_timestep']

    data = (X_list, y_list, f_list, y_test_list, total_lml)
    pickle.dump(data, open("data/tasks-{}-timesteps-{}-noise-{}-seed-{}.pickle".format(M, T, ground_sigma, seed), "wb"))
    return X_list, y_list, [], exp_kwargs, y_test_list


