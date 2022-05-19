import gpflow as gpf
from gpflow.utilities import print_summary, set_trainable, to_default_float, read_values, deepcopy
from gpflow import config 
import scipy.stats as sp
import numpy as np
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from .kernels import create_linear, create_period, create_rbf, Interaction
import itertools
from itertools import chain, combinations

'''

'''
def split_data_by_size(X, y, size = 10):
    X_list = []
    y_list = []
    
    # split data by size
    num_chunks =  np.ceil(X.shape[0]/size)
    chunks = np.array_split(np.arange(X.shape[0]), num_chunks)
    for chunk in chunks: 
        X_chunk = X[chunk, :]
        y_chunk = y[chunk, :]
        X_list.append(X_chunk)
        y_list.append(y_chunk)
        
    return X_list, y_list


def get_data_stats(X_list):
    keys = np.array(list(X_list.keys()))
    M = len(np.unique(keys[:, 0]))
    T = np.ones(M, dtype = np.int64)
    for m in range(M): 
        timesteps = keys[keys[:, 0] == m, 1]
        T[m] = len(timesteps)
    return M, T

def phi(X):
    a = tf.reshape((X[:, 9]), [-1, 1])
    bias = tf.reshape(X[:, 5], [-1, 1])
    action_X = tf.math.multiply(a, X[:, :5])
    phi_X = tf.concat([action_X, bias], 1)
    return phi_X

def log_marginal_likelihood(X, y, kernel, noise, mean_function = gpf.mean_functions.Zero(), heartsteps = False):
    K_XX = kernel(X)
    n = to_default_float(tf.shape(y)[0])
    K_XX += to_default_float(tf.eye(n)) * 1e-6 # jitter
    K_XX += to_default_float(tf.eye(n)) *  noise # variance

    if heartsteps: 
        # Linear advantage function
        advantage = gpf.kernels.Linear(active_dims = [0, 1, 2, 3, 4, 5], variance = tf.ones(6)) 
        phi_X = phi(X)
        K_adv = advantage.K(phi_X, phi_X)
        K_XX += K_adv
        
    
    K_inv = tf.linalg.inv(K_XX)
    K_det = tf.linalg.det(K_XX)
    
    # mean function
    mean_f = mean_function(X)
    res_y = y - mean_f

    fit = -0.5 * tf.tensordot(tf.tensordot(tf.transpose(res_y), K_inv, 1), res_y, 1)[0,0]
    if(K_det == 0):
        complexity = -0.5 * tf.math.log(K_det + 1.)
    else:
        complexity = -0.5 * tf.math.log(K_det)

    const = -0.5 * n * to_default_float(tf.math.log(2 * np.pi))
    return fit + complexity + const

def permute_kernels(kernels, max_num = 3):
    subsets = chain.from_iterable(combinations(kernels, r) for r in range(1,max_num + 1))
    final = [gpf.kernels.Sum(subset) for subset in subsets]
    return final

# set batches
def get_batches(X, y, minibatch_size):
    train_dataset = tf.data.Dataset. \
        from_tensor_slices((X, y)). \
        repeat(). \
        shuffle(y.shape[0])
    train_iter = iter(train_dataset.batch(minibatch_size))

    return train_iter


'''
def initialize_base_kernels_con(X, weak_prior = True, sd = 1., scaling_parameter = True, time_columns = []):
    base_kernels = []
    linear = create_linear(X[:, dim], active_dims = [dim], weak_prior = weak_prior, sd = sd) # linear
    periodic = create_period(X[:, dim], active_dims = [dim], weak_prior = weak_prior, sd = sd) # periodic
    rbf = create_rbf(X[:, dim], active_dims = [dim], weak_prior = weak_prior, sd = sd) # rbf

    # set scaling parameters on or off
    set_trainable(linear.variance, scaling_parameter)
    set_trainable(rbf.variance, scaling_parameter)
    set_trainable(periodic.variance, scaling_parameter)
    if not scaling_parameter: 
        linear.variance.assign(to_default_float(1))
        rbf.variance.assign(to_default_float(1))
        periodic.variance.assign(to_default_float(1))
    
    base_kernels.append(linear)
    if dim in time_columns:
        base_kernels.append(periodic)
    base_kernels.append(rbf)
    
    return(base_kernels)
'''
def kernel_to_string_full(kernel):
    if isinstance(kernel, gpf.kernels.base.Sum):
        kernel_strings = [kernel_to_string_full(subkernel) for subkernel in kernel.kernels]
        return " (" + '+'.join(kernel_strings) + ") "
    elif isinstance(kernel, gpf.kernels.base.Product) or isinstance(kernel, Interaction):
        kernel_strings = [kernel_to_string_full(subkernel) for subkernel in kernel.kernels]
        return " (" + 'x'.join(kernel_strings) + ") "
    else:
        return str(type(kernel).__name__)  
        #return str(type(kernel).__name__)
