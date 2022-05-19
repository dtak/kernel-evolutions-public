"""
The follows kernels contains some features:

    - Linear kernel has parameter location
    - kernel is added parameterization
"""

import itertools

import numpy as np
from gpflow import Parameter
from gpflow.utilities import to_default_float, set_trainable, print_summary, positive
from gpflow.kernels import (Linear as Linear_gpflow,
                            RBF,
                            Kernel,
                            Periodic as Periodic_gpflow,
                            Product)
import gpflow as gpf
from scipy.stats import truncnorm, loguniform
from tensorflow_probability import distributions as tfd
import pandas as pd
import tensorflow as tf

def kernel_to_string(kernel):
    if isinstance(kernel, gpf.kernels.base.Sum):
        kernel_strings = [kernel_to_string(subkernel) for subkernel in kernel.kernels]
        string = " (" + '+'.join(kernel_strings) + ") "
        # remove last parenthesis
        split = string.split(")")
        if(len(split) > 1):
            del split[-1]
        string = ')'.join(split)

        # remove first parenthesis
        split = string.split("(")
        if(len(split) > 1):
            del split[0]
        string = '('.join(split)

        string = string.replace("Linear", "LIN")
        string = string.replace("Periodic", "PER")
        string = string.replace("SquaredExponential", "SE")
        return string.strip()
    elif isinstance(kernel, gpf.kernels.base.Product) or isinstance(kernel, Interaction):
        kernel_strings = [kernel_to_string(subkernel) for subkernel in kernel.kernels]
        string = " (" + 'x'.join(kernel_strings) + ") "
        return string
    else:
        string = str(type(kernel).__name__) + str(kernel.active_dims[0]) 

        return string

class Interaction(Kernel):
    def __init__(self, kernels, variance = 1.):
        super().__init__()
        self.kernels = kernels

        self.product = Product(kernels)
        self.variance = Parameter(variance, transform=positive())
        #for kernel in kernels: 
        #    kernel.variance.assign(1.)
        #    set_trainable(kernel.variance, False)

    def K(self, X, X2=None):
        if X2 is None:
            return self.product.K(X) * self.variance
        else:
            return self.product.K(X, X2) * self.variance

    def K_diag(self, X):
        return self.product.K_diag(X) * self.variance

class Periodic(Periodic_gpflow):
    def __init__(self, base_kernel, variance = 1., period=1., lengthscales = 1.):
        super().__init__(base_kernel, period = period)
        self.variance = base_kernel.variance
        self.lengthscales = base_kernel.lengthscales
        self.lengthscales.assign(lengthscales)
        self.variance.assign(variance)

    def K(self, X, X2=None):
        if X2 is None: 
            return super().K(X)
        else: 
            return super().K(X, X2)
    
    def K_diag(self, X):
        return super().K_diag(X) 

class Linear(Linear_gpflow):
    def __init__(self, variance=1., location=0., bound=None, active_dims=None, rescale = False):
        super().__init__(variance=variance, active_dims=active_dims)
        self.rescale = rescale
        if bound is None:
            self.location = Parameter(location)
        else:
            raise NotImplementedError

    @property
    def ard(self):
        return self.location.shape.ndims > 0

    def K(self, X, X2=None):
        X_shifted = X - self.location
        if X2 is None:
            if self.rescale:
                kernel = super().K(X_shifted, None)
                const = tf.maximum(tf.reduce_max(kernel), 1)/self.variance
                return kernel/const
            else: 
                return super().K(X_shifted, None)
        else:
            X2_shifted = X2 - self.location
            if self.rescale:
                kernel = super().K(X_shifted,X2_shifted)
                train_kernel = super().K(X2_shifted,X2_shifted)
                max_val = tf.maximum(tf.reduce_max(train_kernel), tf.reduce_max(kernel))
                const = tf.maximum(max_val, 1)/self.variance
                return kernel/const
            else: 
                return super().K(X_shifted, X2_shifted)

    def K_diag(self, X):
        X_shifted = X - self.location
        if self.rescale: 
            kernel = super().K(X_shifted, None)
            const = tf.maximum(tf.reduce_max(kernel), 1)/self.variance
            return kernel/const
        else: 
            return super().K_diag(X_shifted)

def create_linear(X, sd=1., active_dims=None, rescale = False, hyper_priors = None):
    kernel = Linear(active_dims = active_dims, rescale = rescale)

    if hyper_priors is None: 
        location = to_default_float(0)
        variance = to_default_float(1)
    else: 
        location_dist = tfd.Normal(to_default_float(hyper_priors['location']['mean']), to_default_float(hyper_priors['location']['var']))
        variance_dist = tfd.LogNormal(to_default_float(hyper_priors['amplitude']['mean']), to_default_float(hyper_priors['amplitude']['var']))
    
        location = location_dist.sample()
        variance = variance_dist.sample()
        
        kernel.location.prior = location_dist
        kernel.variance.prior = variance_dist
    
    kernel.variance = Parameter(variance, transform=positive(lower = 1e-5))
    kernel.location.assign(location)

    return kernel

def create_rbf(X, sd=1., active_dims=None, hyper_priors = None):
    kernel = RBF(active_dims=active_dims)

    if hyper_priors is None: 
        lengthscale = to_default_float(1)
        variance = to_default_float(1)
    else: 
        lengthscale_dist = tfd.LogNormal(to_default_float(hyper_priors['lengthscale']['mean']), to_default_float(hyper_priors['lengthscale']['var']))
        variance_dist = tfd.LogNormal(to_default_float(hyper_priors['amplitude']['mean']), to_default_float(hyper_priors['amplitude']['var']))

        lengthscale = lengthscale_dist.sample()
        variance = variance_dist.sample()
        
        kernel.lengthscales.prior = lengthscale_dist # lengthscale
        kernel.variance.prior = variance_dist

    kernel.lengthscales = Parameter(lengthscale, transform=positive(lower = 1e-5))
    kernel.variance = Parameter(variance, transform=positive(lower = 1e-5))

    return kernel

def create_period(X, sd=1., active_dims=None, hyper_priors = None):
    base_kernel = create_rbf(X, active_dims=active_dims, hyper_priors = hyper_priors)
    kernel = Periodic(base_kernel)
    
    if hyper_priors is None: 
        period = to_default_float(1)
    else: 
        period_dist = tfd.LogNormal(to_default_float(hyper_priors['period']['mean']), to_default_float(hyper_priors['period']['var']))
        period = period_dist.sample()
        kernel.period.prior = period_dist
        
    kernel.period = Parameter(period, transform=positive(lower = 1e-5))

    return kernel
