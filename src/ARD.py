import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from gpflow.utilities import to_default_float, print_summary, deepcopy, positive
import gpflow as gpf
class ARD: 
    def __init__(self, X, y, hyper_priors = None):
        self.X = X
        self.y = y

        self.hyper_priors = hyper_priors
        
        self.D = X.shape[1]
        self.k = gpf.kernels.RBF(lengthscales = tf.ones(self.D))
   
        # Prior distributions
        self.noise_dist = tfd.LogNormal(tf.cast(hyper_priors['noise']['mean'], 'float64'), tf.cast(hyper_priors['noise']['var'], 'float64'))
        self.amplitude_dist = tfd.LogNormal(tf.cast(hyper_priors['amplitude']['mean'], 'float64'), tf.cast(hyper_priors['amplitude']['var'], 'float64'))
        self.lengthscale_dist = tfd.LogNormal(tf.cast(hyper_priors['lengthscale']['mean'], 'float64'), tf.cast(hyper_priors['lengthscale']['var'], 'float64'))
        
        # Track hypers
        self.original_noise = self.noise_dist.sample()
        self.original_lengthscales = np.array([self.lengthscale_dist.sample() for l in range(self.D)])
        self.original_amp = self.amplitude_dist.sample()
                
        if self.original_noise <= 1e-2: 
            self.original_noise = 1e-1
        if self.original_amp <= 1e-4:
            self.original_amp = 1e-1
        if np.all(self.original_lengthscales > 1e-4) == False:
            self.original_lengthscales[self.original_lengthscales <= 1e-4] = 1e-1 
        

    def update_data(self, X, y):
        self.X = X
        self.y = y

    def optimize(self, n_restarts = 10, hyper_proposal_variance = 0.1): 
        normal  = tfd.Normal(0, tf.cast(hyper_proposal_variance, 'float64'))
        
        # Initialize gp with new data
        gp = gpf.models.GPR((self.X, self.y), self.k)

        # Initial hyper values
        max_likelihood = np.NINF
        max_noise = np.copy(gp.likelihood.variance.numpy())
        max_amp = np.copy(gp.kernel.variance.numpy())
        max_lengthscales = np.copy(gp.kernel.lengthscales.numpy())

        # Make sure all values are numerically valid
        if max_noise > 1e-2: 
            gp.likelihood.variance = gpf.Parameter(max_noise, transform = positive(lower = 1e-2))
        else: 
            gp.likelihood.variance = gpf.Parameter(1e-1, transform = positive(lower = 1e-2))
        if np.all(max_lengthscales > 1e-4):
            gp.kernel.lengthscales = gpf.Parameter(max_lengthscales, transform = positive(lower = 1e-4))
        else: 
            max_lengthscales[max_lengthscales <= 1e-4] = 1e-1 
            gp.kernel.lengthscales = gpf.Parameter(max_lengthscales, transform = positive(lower = 1e-4))
        if max_amp > 1e-4: 
            gp.kernel.variance = gpf.Parameter(max_amp, transform = positive(lower = 1e-4))
        else: 
            gp.kernel.variance = gpf.Parameter(1e-3, transform = positive(lower = 1e-4))

        # Set priors
        gp.likelihood.variance.prior = self.noise_dist
        gp.kernel.lengthscales.prior = self.lengthscale_dist
        gp.kernel.variance.prior = self.amplitude_dist

        # Optimize!
        opt = gpf.optimizers.Scipy()
        for i in range(n_restarts):
            # Set random restart values
            if i == 0:
                gp.likelihood.variance.assign(self.original_noise)
                gp.kernel.variance.assign(self.original_amp)
                gp.kernel.lengthscales.assign(self.original_lengthscales)
            else: # start near previous optima
                noise = tf.math.exp(tf.math.log(self.original_noise) + normal.sample())
                if noise > 1e-2:
                    gp.likelihood.variance.assign(noise)
                amp = tf.math.exp(tf.math.log(self.original_amp) + normal.sample())
                if amp > 1e-4:
                    gp.kernel.variance.assign(amp)
                lengthscales = np.array([tf.math.exp(tf.math.log(self.original_lengthscales[i]) + normal.sample()) for i in range(self.D)])
                if np.all(lengthscales > 1e-4) == False:
                    lengthscales[lengthscales <= 1e-4] = 1e-1 
                gp.kernel.lengthscales.assign(lengthscales)              
            
            previous_likelihood = -1. * gp.training_loss()
            try: 
                opt.minimize(gp.training_loss, gp.trainable_variables)
                current_likelihood = -1. * gp.training_loss()
            except: 
                # Revert
                gp.likelihood.variance.assign(self.original_noise + 1e-2)
                gp.kernel.variance.assign(self.original_amp)
                gp.kernel.lengthscales.assign(self.original_lengthscales)
                current_likelihood = previous_likelihood

            if current_likelihood > max_likelihood:
                max_likelihood = current_likelihood
                max_noise = np.copy(gp.likelihood.variance.numpy())
                max_lengthscales = np.copy(gp.kernel.lengthscales.numpy())
                max_amp = np.copy(gp.kernel.variance.numpy())

            print("optimized from {} to {}: noise {}, lengthscales {}, amp {}".format(previous_likelihood, current_likelihood, gp.likelihood.variance.numpy(), gp.kernel.lengthscales.numpy(), gp.kernel.variance.numpy()))
       
        # Set to the max
        self.original_noise = np.copy(max_noise) 
        self.original_lengthscales = np.copy(max_lengthscales)
        self.original_amp = np.copy(max_amp)

