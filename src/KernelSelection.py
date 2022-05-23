import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow as gpf
from gpflow.utilities import print_summary, set_trainable, to_default_float, deepcopy, positive
from itertools import groupby
from operator import itemgetter
from gpflow import Parameter
from scipy.stats import multivariate_normal

from .tools.kernels import create_linear, create_period, create_rbf, Interaction, Linear, Periodic, kernel_to_string

def log_marginal_likelihood(X, y, model, mean_function = gpf.mean_functions.Zero(), heartsteps = False):
    kernel = model['structure']
    noise = model['noise']
    n = to_default_float(tf.shape(y)[0])
    K_XX = kernel(X)
    K_XX += to_default_float(tf.eye(n)) * 1e-6 # jitter
    K_XX += to_default_float(tf.eye(n)) *  noise # variance

    if heartsteps: 
        # Linear advantage function
        advantage = gpf.kernels.Linear(active_dims = [0, 1, 2, 3, 4, 5], variance = tf.ones(6)) 
        phi_X = phi(X)
        K_adv = advantage(phi_X, phi_X)
        K_XX += K_adv
        
    
    K_inv = tf.linalg.inv(K_XX)
    
    # mean function
    mean_f = to_default_float(mean_function(X))
    res_y = y - mean_f

    fit = -0.5 * tf.tensordot(tf.tensordot(tf.transpose(res_y), K_inv, 1), res_y, 1)[0,0]
    
    sign, abs_log_det = tf.linalg.slogdet(K_XX)
    complexity = -0.5 * abs_log_det

    const = -0.5 * n * to_default_float(tf.math.log(2 * np.pi))
        
    #print("likelihood {},  fit {}, complexity {}, const {}".format(fit + complexity + const, fit, complexity, const))
    return fit + complexity + const 

from constants import HYPER_PRIORS_SCALED as HYPER_PRIORS
def MAP(X, y, model, base_dist, heartsteps = False, mean_function = gpf.mean_functions.Zero()):
    # Initialize model
    if heartsteps: 
        gp = HSGP((X, to_default_float(y)), deepcopy(model['structure']), noise = model['noise'])
        set_trainable(gp.advantage, False)
    else: 
        gp = gpf.models.GPR((to_default_float(X), to_default_float(y)), deepcopy(model['structure']))

    # Optimize hypers
    if model['noise'] <= 1e-2: 
        gp.likelihood.variance = Parameter(0.1, transform = positive(lower = 1e-2)) 
    else:
        gp.likelihood.variance = Parameter(model['noise'], transform = positive(lower = 1e-2)) 

    # Set hyper priors
    kernel_hypers, kernel_hyper_types, kernel_active_dims = base_dist.get_hypers(gp.kernel)
    base_dist.hyper_priors = {key:item for key, item in HYPER_PRIORS.items()} # fix base dist hyper prior
    hyper_priors, noise_dist = base_dist.get_hyper_priors(kernel_hyper_types)
    for i, hyper in enumerate(kernel_hypers):
        hyper.prior = hyper_priors[i]
    
    gp.likelihood.variance.prior = noise_dist

    opt = gpf.optimizers.Scipy()
    try: 
        opt.minimize(gp.training_loss, gp.trainable_variables)
    except: 
        print("DECOMP ERROR!")
        print_summary(gp)
        # Revert to original
        #gp.kernel = model['structure']
        #gp.likelihood.variance.assign(model['noise'])
    # Consider model prior
   # posterior = -1 * gp.training_loss()
    #print_summary(gp)
    # Cleanup
    del opt
    del hyper_priors
    del noise_dist

#    print("MAP: ", prior + likelihood)
    opt_model = {"structure": deepcopy(gp.kernel), "noise":np.copy(gp.likelihood.variance.numpy())}
    composition_prior = base_dist.log_prob(opt_model)
    return opt_model, tf.squeeze(composition_prior + log_marginal_likelihood(X, y, opt_model, heartsteps = heartsteps, mean_function = mean_function)).numpy()

def BIC(X, y, model, base_dist, heartsteps = False, mean_function = gpf.mean_functions.Zero(), n_restarts = 0):
    if n_restarts > 0: 
        # Random restarts
        opt_models = []
        likelihoods = []
        for i in range(n_restarts): 
            restart_model = {'noise':np.copy(model['noise']), 'structure':deepcopy(model['structure'])}
            hypers, hyper_types, active_dims = base_dist.get_hypers(restart_model['structure'])
            for hyper in hypers: 
                new_hyper = np.exp(np.log(hyper.numpy() + 1e-7)  + np.random.normal(0,0.1))
                if new_hyper> 1e-2: 
                    hyper.assign(new_hyper)
            new_noise =  np.exp(np.log(restart_model['noise'] + 1e-7)  + np.random.normal(0,0.1))
            if new_noise >1e-2: 
                restart_model['noise'] = new_noise
            opt_model, map_likelihood = MAP(X, y, restart_model, base_dist, heartsteps = heartsteps, mean_function = mean_function)
            del restart_model
            opt_models.append(opt_model)
            likelihoods.append(map_likelihood)
            
        # Get minimum
        max_ind = np.argmax(likelihoods)
        map_likelihood = likelihoods[max_ind]
        opt_model = deepcopy(opt_models[max_ind])
        
        # cleanup
        del likelihoods
        del opt_models
    else: 
        opt_model, map_likelihood = MAP(X, y, model, base_dist, heartsteps = heartsteps, mean_function = mean_function)
    #print_summary(opt_model)

    # Calculate BIC
    umbrella = (len(opt_model['structure'].trainable_variables) + 1) * np.log(X.shape[0]) 
    max_likelihood = 2 * map_likelihood
    bic = umbrella - max_likelihood
    return opt_model, -1 * bic

def phi(X):
    a = tf.reshape((X[:, 9]), [-1, 1])
    bias = tf.reshape(X[:, 5], [-1, 1])
    action_X = tf.math.multiply(a, X[:, :5])
    phi_X = tf.concat([action_X, bias], 1)
    return phi_X


#def diag_ll(vec, mean, noise): 
def diag_ll(vec, mean, noise): 
    k = vec.shape[0]
    diff = vec - mean
    inv = tf.eye(k, dtype =tf.dtypes.float64)/noise
    exp_term = -0.5 * tf.tensordot(tf.tensordot(tf.transpose(diff), inv, 1), diff, 1)
    denom_term = 0.5 * k * (tf.math.log(tf.constant(2. * np.pi, dtype = tf.dtypes.float64)) + tf.math.log(noise))
    return exp_term - denom_term

def approx_lml(X, y, model, mean_function = gpf.mean_functions.Zero(), heartsteps = False, N = 1):
    if heartsteps:
        gp = HSGP((X, y), model['structure'], noise = model['noise'], mean_function = mean_function)
    else:
        gp = gpf.models.GPR((X, y), model['structure'], mean_function = mean_function)

    samples = gp.predict_f_samples(X, N)
    likelihoods = tf.map_fn(lambda x: diag_ll(y, x, model['noise']), samples)
    
    #other_likelihoods = [multivariate_normal.logpdf(tf.squeeze(y), mean = samples[i, :, 0], cov = np.eye(samples.shape[1]) * model['noise']) for i in range(N)]
    return tf.math.reduce_mean(likelihoods)



def bic(X, y, model, mean_function = gpf.mean_functions.Zero(), heartsteps = False):
    k = len(kernel.trainable_variables)
    lml = log_marginal_likelihood(X, y, model, mean_function = mean_function, heartsteps = heartsteps)
    return (2 * lml) - (k * np.log(X.shape[0]))

class MultinomialKernels:
    def __init__(self, components = None, p = None, hyper_priors = None, n_dimensions = None):
        if len(components) != len(p): # check lengths 
            raise ValueError("length mismatch, you must have a probability for each kernel")

        self.kernels = np.array(components)
        self.p = np.array(p)
        self.n = len(self.kernels)
        self.num_dimensions = n_dimensions
        # get string name of each composition
        self.kernel_probabilities = {} # mapping from kernel name to probability
        self.kernel_names = np.array([kernel_to_string(kernel) for kernel in self.kernels])
    
        self.hyper_priors = {key:item for key, item in hyper_priors.items()}
        self.hyper_map = {'lengthscale':0, 'amplitude':1, 'location':2, 'period':3}
    
    def get_hypers(self, kernels):
        '''
        returns a list of references to kernel hyperparameters and a list of string types
        returns hypers, hyper_types, active_dims
        '''
        if isinstance(kernels, gpf.kernels.Product) or isinstance(kernels, gpf.kernels.Sum) or isinstance(kernels, Interaction):
            hypers = []
            hyper_names = []
            active_dims = []

            for subkernel in kernels.kernels:
                subhypers, subnames, subdims = self.get_hypers(subkernel)
                hypers.extend(subhypers)
                hyper_names.extend(subnames)
                active_dims.extend(subdims)
            return hypers, hyper_names, active_dims 
        elif isinstance(kernels, gpf.kernels.Periodic) or isinstance(kernels, Periodic): # periodic
            ad = kernels.active_dims
            return [kernels.base_kernel.lengthscales, kernels.base_kernel.variance, kernels.period], ["lengthscale", "amplitude", "period"], [ad, ad, ad]
        elif isinstance(kernels, gpf.kernels.Linear) or isinstance(kernels, Linear): # linear
            ad = kernels.active_dims[0]
            return [kernels.location, kernels.variance], ["location", "amplitude"], [ad, ad]
        elif isinstance(kernels, gpf.kernels.White): 
            return [], [], []
        else:
            ad = kernels.active_dims[0]
            return [kernels.lengthscales, kernels.variance], ["lengthscale", "amplitude"], [ad, ad]

    def set_hypers(self, hypers, hyper_list):
        '''
        assigns hyperparameters to the kernel
        hyper_list: list of hyperparameters for the kernel
        '''
        num_hypers = len(hypers)
        for i, hyper in enumerate(hypers):
            hyper.assign(tf.reshape(hyper_list[i], hyper.shape))


    def get_full_hypers(self, model):
        noise = model['noise']
        kernel = model['structure']
        structure = self.get_structure(model)
        hyper_mat = np.zeros((self.n + 1, 4))
        hyper_mat[-1, -1] = np.log(noise)
        if structure.sum() > 0: 
            # [kernel unit + 1 for noise]  x [lengthscale, amplitude, location, period]
            # get actual hyperparameter values
            hyper_indices = np.where(structure)[0]
            for i, component in enumerate(kernel.kernels):
                kernel_hypers, kernel_hyper_types, kernel_active_dims = self.get_hypers(component)
                index = hyper_indices[i]
                #print("Hypers for {} index {} = {}?".format(kernel_to_string(component), index, self.kernel_names[index]))
                # fill the hyper matrix
                for j, hyper in enumerate(kernel_hypers): 
                    col = self.hyper_map[kernel_hyper_types[j]]
                    if col == 2: 
                        hyper_mat[index, col] = hyper.numpy()
                    else: 
                        hyper_mat[index, col] = np.log(hyper.numpy())

        return hyper_mat

    def to_model(self, structure, hypers, copy = False):
        if structure.sum() > 0: 
            hyper_indices = np.where(structure == 1)[0]
            if copy:
                kernels = [deepcopy(self.kernels[i]) for i in range(self.n) if i in hyper_indices]
            else: 
                kernels = self.kernels[hyper_indices]
            
            for i, component in enumerate(kernels):
                index = hyper_indices[i]
                kernel_hypers, kernel_hyper_types, kernel_active_dims = self.get_hypers(component)
                #print("Hypers for {} index {} = {}?".format(kernel_to_string(component), index, self.kernel_names[index]))
                # set the hypers
                for j, hyper in enumerate(kernel_hypers):
                    col = self.hyper_map[kernel_hyper_types[j]]
                    if col == 2: 
                        hyper.assign(hypers[index, col])
                    else:
                        exponentiated = np.exp(hypers[index, col])
                        if exponentiated > 1e-4:
                            hyper.assign(tf.reshape(exponentiated, hyper.shape))
                        else: 
                            hyper.assign(tf.reshape(1e-3, hyper.shape))
            k = gpf.kernels.Sum(kernels)
        else: 
            k = gpf.kernels.White(variance = 1e-4, active_dims = [0])
        
        exponentiated = np.exp(hypers[-1, -1])
        if exponentiated > 1e-2: 
            noise = exponentiated
        else: 
            noise = 1e-2

        return {'structure':k, 'noise':noise}
    
    def log_prob(self, model):
        kernel = model['structure']
        noise = model['noise']

        # calculate probability of model
        encoding = self.get_structure(model) # break kernel into components
        probs = np.array([self.p[i] if encoding[i] == 1 else 1 - self.p[i] for i in range(self.n)] )
        probs = probs + 1e-7

        # calculate probability of hypers
        '''
        kernel_hypers, kernel_hyper_types, kernel_active_dims = self.get_hypers(kernel)
        hyper_dists, noise_dist = self.get_hyper_priors(kernel_hyper_types)
        hyper_probs = 0
        for i, hyper in enumerate(kernel_hypers): 
            hyper_probs += hyper_dists[i].log_prob(hyper.numpy().flatten())
        '''
        if encoding.sum() > 0: 
            kernel_hypers, kernel_hyper_types, kernel_active_dims = self.get_hypers(kernel)
            hyper_dists, noise_dist = self.get_hyper_priors(kernel_hyper_types)
            hyper_probs = 0
            for i, hyper in enumerate(kernel_hypers):
                #print(kernel_hyper_types[i], hyper_dists[i].log_prob(hyper.numpy().flatten()))
                hyper_probs += hyper_dists[i].log_prob(hyper.numpy().flatten())
        else: 
            hyper_probs = 0
            _, noise_dist = self.get_hyper_priors([])
        #print(noise)
        #print("kernel prob {} hyper prob {} noise prob {}".format(np.sum(np.log(probs)), hyper_probs,  noise_dist.log_prob(noise).numpy()))
        
        return np.sum(np.log(probs)) + hyper_probs + noise_dist.log_prob(noise).numpy()


    def get_structure(self, model):
        '''
        kernel_components = [kernel_to_string(k) for k in model['structure'].kernels]
        encoding = [1 if k in kernel_components else 0 for k in self.kernel_names]
        return np.array(encoding)
        '''
        if isinstance(model['structure'], gpf.kernels.Sum): 
            kernel_components = [kernel_to_string(k) for k in model['structure'].kernels]
            encoding = [1 if k in kernel_components else 0 for k in self.kernel_names]
            return np.array(encoding)
        else: 
            return np.zeros(self.kernels.shape[0])

    def to_string(self, model): 
        vec = self.get_structure(model)
        kernels = self.kernel_names[vec > 0]
        return("+".join(kernels))

    def sample(self):
        kernels = [deepcopy(self.kernels[i]) for i in range(len(self.kernels)) if np.random.rand() <= self.p[i]]
        if len(kernels) == 0: # TODO: this might be a problem... can we just get rid of the 0 kernel scenario? 
            hypers = self.sample_hypers([])
            kernels = [gpf.kernels.White(variance = 1e-4, active_dims = [0])]
            '''
            kernels = [deepcopy(np.random.choice(self.kernels))]
            '''

        kernel = gpf.kernels.Sum(kernels)
        
        # sample hypers
        kernel_hypers, kernel_hyper_types, kernel_active_dims = self.get_hypers(kernel)
        hypers = self.sample_hypers(kernel_hyper_types)
        self.set_hypers(kernel_hypers, hypers[:-1])
        #print("sampling with probs {}".format(self.p))
        #print(kernel_to_string(kernel))
        return {'structure':kernel, 'noise':hypers[-1]}
    
    def get_hyper_priors(self, hyper_types):
        '''
        could refactor to taking up less space
        '''
        distributions = []
        # distribution over hypers
        for i, hyper_type in enumerate(hyper_types):
            if hyper_type == "location":
                if 'location' in self.hyper_priors: 
                    dist = self.hyper_priors['location']
                    dist = tfd.Normal(tf.cast(dist['mean'], 'float64'), tf.cast(dist['var'], 'float64'))
                else:
                    dist = tfd.Normal(to_default_float(0), to_default_float(0.5))
            elif hyper_type == "period":
                if 'period' in self.hyper_priors:
                    dist = self.hyper_priors['period']
                    dist = tfd.LogNormal(tf.cast(dist['mean'], 'float64'), tf.cast(dist['var'], 'float64'))
                else: 
                    dist = tfd.LogNormal(to_default_float(0), to_default_float(0.5)) 
            elif hyper_type == "lengthscale": 
                if 'lengthscale' in self.hyper_priors:
                    dist = self.hyper_priors['lengthscale']
                    dist = tfd.LogNormal(tf.cast(dist['mean'], 'float64'), tf.cast(dist['var'], 'float64'))
                else:
                    dist = tfd.LogNormal(to_default_float(0), to_default_float(0.5)) 
            elif hyper_type == "amplitude": 
                if 'amplitude' in self.hyper_priors:
                    dist = self.hyper_priors['amplitude']
                    dist = tfd.LogNormal(tf.cast(dist['mean'], 'float64'), tf.cast(dist['var'], 'float64'))
                else:
                    dist = tfd.LogNormal(to_default_float(0), to_default_float(0.5))
            distributions.append(dist)

        # distribution over noise
        if 'noise' in self.hyper_priors:
            dist = self.hyper_priors['noise']
            noise_dist = tfd.LogNormal(tf.cast(dist['mean'], 'float64'), tf.cast(dist['var'], 'float64'))
        else:
            noise_dist = tfd.LogNormal(to_default_float(0), to_default_float(0.5))
        
        return distributions, noise_dist

    def sample_hypers(self, kernel_hyper_types):
        hypers = []
        distributions, noise_dist = self.get_hyper_priors(kernel_hyper_types)
        
        for i, hyper_type in enumerate(kernel_hyper_types):
            hypers.append(distributions[i].sample().numpy())

        hypers.append(noise_dist.sample().numpy())
        return np.array(hypers)


