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

def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        try: 
            return float(s)
        except ValueError: 
            if s == "True" or s == "False": 
                return bool(s)
            else: 
                return s


def dir_to_args(outdir):
    exp_dir = outdir.split("/")[-1] # last one is output
    parts = exp_dir.split("-")
    keys = parts[0:len(parts):2]
    vals = [int_or_float(part) for part in parts[1:len(parts):2]]

    args = dict(zip(keys, vals))
    return args

def get_path_from_setting(method_name, exp_kwargs, users, outdir): 

    is_singletask = "shrinkage" in method_name or "cks" in method_name or "mh" in method_name or "ard" in method_name
    
    # Path differs depending on multitask or single task
    if is_singletask:
        # pure selection method runs only for specified user
        user_ind = exp_kwargs['user']
        if "toy" in method_name or "multi" in method_name: 
            user = user_ind
        else: 
            user = users[user_ind]
        del exp_kwargs['user']
        
        outdir += args_to_dir(exp_kwargs)
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

    # Stuff you don't want to pass to method
    del exp_kwargs['data_seed']
    if 'timestep_min' in exp_kwargs:
        del exp_kwargs['timestep_min']
    if 'M' in exp_kwargs:
        del exp_kwargs['M']
    if 'chunk_size' in exp_kwargs:
        del exp_kwargs['chunk_size']
    if 'noise' in exp_kwargs: 
        del exp_kwargs['noise']
    if 'ground_kernel' in exp_kwargs: 
        del exp_kwargs['ground_kernel']
    if 'num_strong_features' in exp_kwargs: 
        del exp_kwargs['num_strong_features']
    if 'num_weak_features' in exp_kwargs: 
        del exp_kwargs['num_weak_features']
    if 'num_features' in exp_kwargs: 
        del exp_kwargs['num_features']
    return outdir, exp_kwargs

def get_timeseries_data(method_name, exp_kwargs, data_dir = None): 
    seed = exp_kwargs['data_seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    
    # Read data
    df = pd.read_csv(data_dir + "df.csv")
    
    # Sample users
    M = exp_kwargs['M']
    users = np.random.choice(df.drop(['t', 'Date'], axis = 1).columns, M, replace = False)
    print("Using {}  users {}".format(users.shape[0], users))
    
    # Data from each task
    X_list = {}
    y_list = {}
    for m, user in enumerate(users):
        # get data
        X_user = np.arange(df.shape[0]).reshape(-1, 1)
        y_user = df[user].values.reshape(-1, 1)
    
        # Split by column which denotes time
        X_user_list = []
        y_user_list = []
        times = np.unique(df['t'])
        times = np.sort(times)
        for time in times:
            time_indices = df['t'] == time
            X_user_list.append(X_user[time_indices, :])
            y_user_list.append(y_user[time_indices, :])

        T = len(y_user_list)
        
#        fig, ax = plt.subplots(T, 1, figsize = (10, 30), sharex = True)
        # for each timestep
        for t in range(T):
            # training is CUMULATIVE set
            X_train = np.vstack(X_user_list[:(t+1)])
            y_train = np.vstack(y_user_list[:(t+1)])
            print("({}, {}) shape: {}".format(m, t, X_train.shape))
            X_list[(m, t)] = to_default_float(X_train )
            y_list[(m, t)] = y_train
#            ax[t].plot(X_train, y_train)
#        plt.suptitle(m)
#        plt.show()

    return X_list, y_list, users, exp_kwargs


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

def get_multi_data(method_name, exp_kwargs): 
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

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Generate data
    X_list = {}
    y_list = {}
    y_test_list = []
    X_test = np.linspace(0, 20, 200).reshape(-1, 1)
    X_test = np.hstack([X_test, np.random.uniform(0, 20, (200, 2))])
    T = len(num_per_timestep)
    zero_centered = (np.zeros((1, 1)), np.zeros((1, 1)))
    f_list = {}

    fig, ax = plt.subplots(M, T, figsize = (10, 30), sharex = True, sharey = False)
    # Show data
    total_lml = 0
    for m in range(M): 
        if m < M//2:
            periodic_kernel = sample_periodic(6, 5, 1)
            periodic_kernel.base_kernel.active_dims = [0]
            linear_kernel = sample_linear(4)
            linear_kernel.active_dims = [0]

            gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
        else:
            rbf_kernel = sample_rbf(1, 10)
            rbf_kernel.active_dims = [0]
            gp = gpf.models.GPR(zero_centered, rbf_kernel)
        
        gp.likelihood.variance.assign(ground_sigma)
        gpf.utilities.print_summary(gp)
        
        # sample a function from the GP
        X_train = np.random.uniform(0, 20, (np.sum(num_per_timestep), 3))
        X = np.vstack((X_train, X_test))
        f = gp.predict_f_samples(to_default_float(X[:, 0].reshape(-1, 1))) 
        y = f + np.random.normal(0, ground_sigma, size = f.shape)
        y_test_list.append(y[(-1 * X_test.shape[0]):, :])
        f_list[m] = f

        # timesteps
        for t in range(T): 
            num_points = np.sum(num_per_timestep[:(t + 1)])
            X_total = X[:num_points, :]
            y_total = y[:num_points, :]

            print(X_total.shape)
            
            X_list[(m, t)] = X_total
            y_list[(m, t)] = y_total
        
            total_lml += log_marginal_likelihood(X[:num_points, :], y[:num_points, :], {'structure': gp.kernel, 'noise': ground_sigma})

            upto = int(np.sum(num_per_timestep[:t]))
            
    #        ax[m][t].plot(X_total[:upto, 0], y_total[:upto], "o", color = "grey", alpha = 0.7, markersize=3) # past points
    #        ax[m][t].plot(X_total[upto:, 0], y_total[upto:], 'o', color = "red", markersize =3) 
    #        ax[m][t].plot(X_total[:, 1], y_total, 'o', color = "blue", markersize =3) 
    #        ax[m][t].plot(X[-200:, 0], f[-200:,0], color = "black") 
    print("GROUND TRUTH LML: {}".format(total_lml))
    #plt.show()
    if 'num_per_timestep' in exp_kwargs: 
        del exp_kwargs['num_per_timestep']

    data = (X_list, y_list, f_list, y_test_list, total_lml)
    pickle.dump(data, open("data/kernel-selection-toy/multi-tasks-{}-timesteps-{}-noise-{}-seed-{}.pickle".format(M, T, ground_sigma, seed), "wb"))
    return X_list, y_list, [], exp_kwargs, y_test_list

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

    # Ground truth kernels
    ground_kernel_1 = gpf.kernels.Linear(variance = 1.) + gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1., variance = 5.), period = 4.)
    ground_kernel_2 = gpf.kernels.RBF(variance = 5)
    ground_kernel_3 = gpf.kernels.RBF(variance = 5, lengthscales = 4.)  *  gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1., variance = 3.), period = 2.) + gpf.kernels.RBF(variance = 10., lengthscales = 7.)
    ground_kernel_4 = gpf.kernels.Linear(variance = 1.) + gpf.kernels.RBF(lengthscales = 2, variance = 2)
    ground_kernel_5 = gpf.kernels.RBF(variance = 10, lengthscales = 5) * gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1, variance = 5), period = 2)

    # Generate data
    X_list = {}
    y_list = {}
    y_test_list = []
    X_test = np.linspace(0, 20, 200).reshape(-1, 1)
    T = len(num_per_timestep)
    zero_centered = to_default_float((np.zeros((1, 1)), np.zeros((1, 1))))
    f_list = {}

    #fig, ax = plt.subplots(M, T, figsize = (10, 30), sharex = True, sharey = False)
    # Show data
    total_lml = 0
    for m in range(M): 
        if ground_kernel == "sim2":
            if m < M//2: # LIN + PER
                periodic_kernel = sample_periodic(6, 5, 1)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
                #rbf_kernel = sample_rbf(1, 5)
                #gp = gpf.models.GPR(zero_centered, rbf_kernel)
            else: # LIN + SE
                #rbf_kernel = sample_rbf(1, 3)
                #linear_kernel = sample_linear(4)
                #gp = gpf.models.GPR(zero_centered, linear_kernel + rbf_kernel)
                periodic_kernel = sample_periodic(3, 5, 1)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
        elif ground_kernel == "sim1": 
            if m < M//2: # LIN + PER
                periodic_kernel = sample_periodic(4, 5, 1)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + periodic_kernel)
            else: # LIN + SE
                rbf_kernel = sample_rbf(3, 5)
                linear_kernel = sample_linear(4)
                gp = gpf.models.GPR(zero_centered, linear_kernel + rbf_kernel)
                #periodic_kernel = sample_periodic(5, 3, 1)
                #rbf_kernel = sample_rbf(10, 10)
                #gp = gpf.models.GPR(zero_centered, periodic_kernel * rbf_kernel)
                
                #periodic_kernel = sample_periodic(5, 5, 1)
                #rbf_kernel = sample_rbf(5, 10)
                #gp = gpf.models.GPR(zero_centered, rbf_kernel + periodic_kernel)
        elif ground_kernel == "complex":
            '''
            periodic_kernel = sample_periodic(2, 5, 1)
            rbf_kernel = sample_rbf(5, 4)
            rbf_kernel2 = sample_rbf(10, 7)
            gp = gpf.models.GPR(zero_centered, periodic_kernel * rbf_kernel + rbf_kernel2)
            '''
            if m < M//2: # LIN * PER
                periodic_kernel = sample_periodic(5, 1, 1)
                linear_kernel = sample_linear(4)
#                rbf_kernel = sample_rbf(10, 7)
                gp = gpf.models.GPR(zero_centered, linear_kernel * periodic_kernel)
            else: # SE * PER
                periodic_kernel = sample_periodic(3, 5, 1)
                rbf_kernel = sample_rbf(5, 4)
                rbf_kernel2 = sample_rbf(10, 7)
                gp = gpf.models.GPR(zero_centered, periodic_kernel * rbf_kernel + rbf_kernel2)

                #periodic_kernel = sample_periodic(5, 3, 1)
                #rbf_kernel = sample_rbf(10, 10)
                #gp = gpf.models.GPR(zero_centered, periodic_kernel * rbf_kernel)
        elif ground_kernel == "simpsamp": # sample hypers
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
        gpf.utilities.print_summary(gp)
        
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

 #           print(X_total.shape)
            
            X_list[(m, t)] = X_total
            y_list[(m, t)] = y_total
        
            total_lml += log_marginal_likelihood(X[:num_points, :], y[:num_points, :], {'structure': gp.kernel, 'noise': ground_sigma})

            upto = int(np.sum(num_per_timestep[:t]))
            
     #       ax[m][t].plot(X_total[:upto, :], y_total[:upto], "o", color = "grey", alpha = 0.7, markersize=3) # past points
     #       ax[m][t].plot(X_total[upto:, :], y_total[upto:], 'o', color = "red", markersize =3) 
     #       ax[m][t].plot(X[-200:, :], f[-200:,:], color = "black") 
    print("GROUND TRUTH LML: {}".format(total_lml))
    #plt.show()
    if 'num_per_timestep' in exp_kwargs: 
        del exp_kwargs['num_per_timestep']

    data = (X_list, y_list, f_list, y_test_list, total_lml)
    pickle.dump(data, open("data/kernel-selection-toy/tasks-{}-timesteps-{}-noise-{}-seed-{}.pickle".format(M, T, ground_sigma, seed), "wb"))
    return X_list, y_list, [], exp_kwargs, y_test_list


'''
def get_toy_data(method_name, exp_kwargs): 
    M = exp_kwargs['M']
    ground_sigma = exp_kwargs['noise']
    seed = exp_kwargs['data_seed'] 
    num_per_timestep = exp_kwargs['num_per_timestep']
    ground_kernel = exp_kwargs['ground_kernel']

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Ground truth kernels
    ground_kernel_1 = gpf.kernels.Linear(variance = 1.) + gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1., variance = 5.), period = 4.)
    ground_kernel_2 = gpf.kernels.RBF(variance = 5)
    ground_kernel_3 = gpf.kernels.RBF(variance = 5, lengthscales = 4.)  *  gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1., variance = 3.), period = 2.) + gpf.kernels.RBF(variance = 10., lengthscales = 7.)
    ground_kernel_4 = gpf.kernels.Linear(variance = 1.) + gpf.kernels.RBF(lengthscales = 2, variance = 2)
    ground_kernel_5 = gpf.kernels.RBF(variance = 10, lengthscales = 5) * gpf.kernels.Periodic(gpf.kernels.RBF(lengthscales = 1, variance = 5), period = 2)

    # Generate data
    X_list = {}
    y_list = {}
    y_test_list = []
    X_test = np.linspace(0, 20, 200).reshape(-1, 1)
    T = len(num_per_timestep)
    zero_centered = to_default_float((np.zeros((1, 1)), np.zeros((1, 1))))
    f_list = {}

 #   fig, ax = plt.subplots(M, T, figsize = (10, 30), sharex = True, sharey = False)
    # Show data
    total_lml = 0
    for m in range(M): 
        if ground_kernel == "similar": 
            if m < M//3: # LIN + PER
                gp = gpf.models.GPR(zero_centered, ground_kernel_1)
            elif m >= M//3 and m < M//3 * 2: # LIN + SE
                gp = gpf.models.GPR(zero_centered, ground_kernel_4)
            else: # SE X PER
                gp = gpf.models.GPR(zero_centered, ground_kernel_5)
        elif ground_kernel == "complex": 
            if m < M//2:
                gp = gpf.models.GPR(zero_centered, ground_kernel_1)
            else: 
                gp = gpf.models.GPR(zero_centered, ground_kernel_3)
        elif ground_kernel == "simpsamp": # sample hypers
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
        else:
            if m < M//2:
                gp = gpf.models.GPR(zero_centered, ground_kernel_1)
            else: 
                gp = gpf.models.GPR(zero_centered, ground_kernel_2)
        
#        gpf.utilities.print_summary(gp)
        gp.likelihood.variance.assign(ground_sigma)
        
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

 #           print(X_total.shape)
            
            X_list[(m, t)] = X_total
            y_list[(m, t)] = y_total
        
            total_lml += log_marginal_likelihood(X[:num_points, :], y[:num_points, :], {'structure': gp.kernel, 'noise': ground_sigma})

            upto = int(np.sum(num_per_timestep[:t]))
            
 #           ax[m][t].plot(X_total[:upto, :], y_total[:upto], "o", color = "grey", alpha = 0.7, markersize=3) # past points
 #           ax[m][t].plot(X_total[upto:, :], y_total[upto:], 'o', color = "red", markersize =3) 
#            ax[m][t].plot(X[-200:, :], f[-200:,:], color = "black") 
    print("GROUND TRUTH LML: {}".format(total_lml))
#    plt.show()
    if 'num_per_timestep' in exp_kwargs: 
        del exp_kwargs['num_per_timestep']

    data = (X_list, y_list, f_list, y_test_list, total_lml)
    pickle.dump(data, open("data/kernel-selection-toy/tasks-{}-timesteps-{}-noise-{}-seed-{}.pickle".format(M, T, ground_sigma, seed), "wb"))
    return X_list, y_list, [], exp_kwargs, y_test_list

'''
