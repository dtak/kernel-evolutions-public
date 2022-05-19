from gpflow.utilities import to_default_float, print_summary, deepcopy, positive
import pickle
from src.tools.utils import get_batches, split_data_by_size, get_data_stats
from src.experiments.base_kernels import initialize_base_kernels, polynomial_kernel_expansion
from src.CompositionalKernelSearch import CompositionalKernelSearch
from src.ShrinkageModel import ShrinkageModel
from src.heartsteps.HSGP import HSGP
from constants import HYPER_PRIORS_SCALED, HYPER_PRIORS_TOY, KERNEL_POOL,TOY_INCLUSION_PROB
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow as gpf
import time

from src.KernelSelection import MultinomialKernels
from src.KernelSelection import log_marginal_likelihood as kernel_likelihood
from src.MHSelection import MHSelection


from src.KernelSelection import log_marginal_likelihood
from src.heartsteps.HSGP import HSGP
import gpflow as gpf
from gpflow.utilities import to_default_float, print_summary, deepcopy
import tensorflow as tf
from scipy.stats import multivariate_normal
from src.experiments.experiment_tools import get_data_stats
import time
from constants import HYPER_PRIORS_TOY
from tensorflow_probability import distributions as tfd   
def rmse(kernel, X_train, y_train, X_test, y_test, heartsteps = False):
    noise = kernel['noise']
    if isinstance(kernel['structure'], gpf.kernels.White): 
        if kernel['structure'].variance > kernel['noise']: # white kernel is equivalent to noise
            noise = kernel['structure'].variance.numpy()

    if heartsteps:
        gp = HSGP((X_train, to_default_float(y_train)), kernel['structure'], noise = kernel['noise'])
        mean_f, _ = gp.predict_y(X_test)
    else: 
        n = to_default_float(tf.shape(X_train)[0])
        K_star_X = kernel['structure'](X_test, X_train)
        K_XX = kernel['structure'](X_train)

        K_XX += to_default_float(tf.eye(n)) * noise # noise
        K_inv = tf.linalg.solve(K_XX, to_default_float(tf.eye(n)))
        mean_f = tf.tensordot(tf.tensordot(K_star_X, K_inv, 1), to_default_float(y_train), 1)
    if np.sqrt(np.mean(np.power(y_test - mean_f, 2))) > 100: 
        print(np.sqrt(np.mean(np.power(y_test - mean_f, 2))))
        print_summary(kernel)
        print("")

    return np.sqrt(np.mean(np.power(y_test - mean_f, 2)))

def pdf_multivariate_gauss(x, mean, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    # `eigh` assumes the matrix is Hermitian.
    vals, vecs = np.linalg.eigh(cov)
    logdet     = np.sum(np.log(vals))
    valsinv    = np.array([1./v for v in vals])
    # `vecs` is R times D while `vals` is a R-vector where R is the matrix
    # rank. The asterisk performs element-wise multiplication.
    U          = vecs * np.sqrt(valsinv)
    rank       = len(vals)
    dev        = x - mean
    # "maha" for "Mahalanobis distance".
    maha       = np.square(np.dot(dev, U)).sum()
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)

def posterior_predictive_likelihood(kernel, X_train, y_train, X_test, y_test, heartsteps = False):
    noise = kernel['noise']
   
    if isinstance(kernel['structure'], gpf.kernels.White): 
        if kernel['structure'].variance > kernel['noise']: # white kernel is equivalent to noise
            noise = kernel['structure'].variance.numpy()
            print_summary(kernel)
            print(noise)
    if heartsteps:
        gp = HSGP((X_train, to_default_float(y_train)), kernel['structure'], noise = kernel['noise'])
        mean_f, cov_f = gp.predict_y(X_test, full_cov = True)
    else: 
        n = to_default_float(tf.shape(X_train)[0])
        K_star_X = kernel['structure'](X_test, X_train)
        K_X_star = kernel['structure'](X_train, X_test)
        K_star_star = kernel['structure'](X_test, X_test)
        
        K_XX = kernel['structure'](X_train)
        K_XX += to_default_float(tf.eye(n)) * (noise + 1e-7) # noise
        K_inv = tf.linalg.solve(K_XX, to_default_float(tf.eye(n)))
        mean_f = tf.tensordot(tf.tensordot(K_star_X, K_inv, 1), to_default_float(y_train), 1)
        cov_f = K_star_star - tf.tensordot(tf.tensordot(K_star_X, K_inv, 1), K_X_star, 1)
        cov_f = cov_f.numpy() + (np.eye(mean_f.shape[0]) *  (tf.squeeze(noise).numpy() + 1e-4)) # add observation noise
        '''
        if isinstance(kernel['structure'], gpf.kernels.White): 
            print(cov_f[:2,:2])
            print(mean_f[:2,:])
            print(y_test[:2, :])
            print(pdf_multivariate_gauss(np.array(y_test).flatten()[:2], mean_f.numpy().flatten()[:2], cov_f[:2,:2]) )
           '''

    ll2 = pdf_multivariate_gauss(np.array(y_test).flatten(), mean_f.numpy().flatten(), cov_f) 
    print(ll2)
    #ll1 = multivariate_normal.logpdf(np.array(y_test).flatten(), mean = tf.squeeze(mean_f).numpy(), cov = cov_f)
    return ll2

def ard(X_list, y_list, 
        n_restarts = 10, 
        data = "", 
        train_seed = 0, 
        adapt_noise_prior = False, 
        hyper_proposal_variance = 0.05, 
        prior = False, 
        outdir = "", 
        user = 0
        ):
    print("SEED ", (train_seed * 100) + user)
    np.random.seed((train_seed * 100) + user)
    tf.random.set_seed((train_seed * 100)  + user)

    D = X_list[0].shape[1]
    k = gpf.kernels.RBF(lengthscales = tf.ones(D))
       
    heartsteps = data == "heartsteps"
    if data == "toy" or data == "multi": 
        hyper_priors = HYPER_PRIORS_TOY
    else: 
        hyper_priors = HYPER_PRIORS_SCALED

    # Sample values
    noise_dist = tfd.LogNormal(tf.cast(hyper_priors['noise']['mean'], 'float64'), tf.cast(hyper_priors['noise']['var'], 'float64'))
    amplitude_dist = tfd.LogNormal(tf.cast(hyper_priors['amplitude']['mean'], 'float64'), tf.cast(hyper_priors['amplitude']['var'], 'float64'))
    lengthscale_dist = tfd.LogNormal(tf.cast(hyper_priors['lengthscale']['mean'], 'float64'), tf.cast(hyper_priors['lengthscale']['var'], 'float64'))
    normal  = tfd.Normal(0, tf.cast(hyper_proposal_variance, 'float64'))

    original_noise = None
    original_lengthscales = None
    original_amp = None

    for t in range(len(X_list)):
        print(t)
        X_train = X_list[t]
        y_train = to_default_float(y_list[t])
        
        # Initialize gp with new data
        if data == "heartsteps": 
            gp = HSGP((X_train, y_train), k)
        else: 
            gp = gpf.models.GPR((X_train, y_train), k)

        # Initial hyper values
        max_likelihood = np.NINF
        max_noise = np.copy(gp.likelihood.variance.numpy())
        max_amp = np.copy(gp.kernel.variance.numpy())
        max_lengthscales = np.copy(gp.kernel.lengthscales.numpy())

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

        if prior:
            gp.likelihood.variance.prior = noise_dist
            gp.kernel.lengthscales.prior = lengthscale_dist
            gp.kernel.variance.prior = amplitude_dist

        opt = gpf.optimizers.Scipy()
        start_time = time.time()

        for i in range(n_restarts):
            # Set random restart values
            if t == 0:
                original_noise = noise_dist.sample()
                original_lengthscales = np.array([lengthscale_dist.sample() for l in range(D)])
                original_amp = amplitude_dist.sample()

                if original_noise <= 1e-2: 
                    original_noise = 1e-1
                gp.likelihood.variance.assign(original_noise)

                if original_amp <= 1e-4:
                    original_amp = 1e-1
                gp.kernel.variance.assign(original_amp)

                if np.all(original_lengthscales > 1e-4) == False:
                    original_lengthscales[original_lengthscales <= 1e-4] = 1e-1 
                gp.kernel.lengthscales.assign(original_lengthscales)
            else: # start near previous optima
                noise = tf.math.exp(tf.math.log(original_noise) + normal.sample())
                if noise > 1e-2:
                    gp.likelihood.variance.assign(noise)
                amp = tf.math.exp(tf.math.log(original_amp) + normal.sample())
                if amp > 1e-4:
                    gp.kernel.variance.assign(amp)
                lengthscales = np.array([tf.math.exp(tf.math.log(original_lengthscales[i]) + normal.sample()) for i in range(D)])
                if np.all(lengthscales > 1e-4) == False:
                    lengthscales[lengthscales <= 1e-4] = 1e-1 
                gp.kernel.lengthscales.assign(lengthscales)              
            
            # Optimize
#            print("BEFORE")
#            print_summary(gp)
            previous_likelihood = -1. * gp.training_loss()
            try: 
                opt.minimize(gp.training_loss, gp.trainable_variables)
                current_likelihood = -1. * gp.training_loss()
            except: 
                print("ERROR")
                # Revert
                gp.likelihood.variance.assign(original_noise + 1e-2)
                gp.kernel.variance.assign(original_amp)
                gp.kernel.lengthscales.assign(original_lengthscales)
                current_likelihood = previous_likelihood
#            print("AFTER")
#            print_summary(gp)
            if current_likelihood > max_likelihood:
                max_likelihood = current_likelihood
                max_noise = np.copy(gp.likelihood.variance.numpy())
                max_lengthscales = np.copy(gp.kernel.lengthscales.numpy())
                max_amp = np.copy(gp.kernel.variance.numpy())

            print("optimized from {} to {}: noise {}, lengthscales {}, amp {}".format(previous_likelihood, current_likelihood, gp.likelihood.variance.numpy(), gp.kernel.lengthscales.numpy(), gp.kernel.variance.numpy()))
       
        end_time = time.time()
        runtime = end_time - start_time

        # Set to the max
        print("MAX FOUND {}: noise {}, lengthscales {}, amp {}".format(max_likelihood, max_noise, max_lengthscales, max_amp))
        original_noise = np.copy(max_noise) 
        original_lengthscales = np.copy(max_lengthscales)
        original_amp = np.copy(max_amp)

        # Pickle results
        print("RUNTIME: ", runtime)
        final_kernel = gpf.kernels.RBF(lengthscales = original_lengthscales, variance = original_amp)
        model = {'model':{'structure':final_kernel, 'noise':original_noise}, 
                'X': X_list[t], 'y':y_list[t], 'posterior_likelihood': tf.constant(max_likelihood).numpy(), 
                'likelihood_params':{'heartsteps':False, "mean_function":gpf.mean_functions.Zero()}}
        results = {"model":model, "runtime":runtime}
        with open("{}chunk_{}.pickle".format(outdir, t), 'wb') as handle:
            pickle.dump(results, handle)
        if t < len(X_list) - 1: 
            print("EVALUATING")
            X_test = X_list[t + 1][X_list[t].shape[0]:, :]
            y_test = y_list[t + 1][X_list[t].shape[0]:, :]
            print(X_list[t].shape, X_list[t+1].shape)
            print(rmse({'structure':final_kernel, 'noise':original_noise}, X_list[t], y_list[t], X_test, y_test, heartsteps = heartsteps))


def mh(X_list, y_list, 
        n_iters= 100,
        hyper_proposal_variance = 0.1, 
        burnin = 0.4,
        component_inclusion_probability = 0.05,
        adapt_noise_prior = False,
        interaction = True,
        data = "", 
        user = 0,
        outdir = "",
        train_seed = 0):
    
    np.random.seed((train_seed * 100) + user)
    tf.random.set_seed((train_seed * 100) + user)
#    np.random.seed(train_seed)
#    tf.random.set_seed(train_seed)
    X_all = np.vstack([X_list[t] for t in range(len(X_list))])

    heartsteps = data == "heartsteps" # see if heartsteps
    if data == "toy" or data == "multi": 
        base_kernels = initialize_base_kernels(X_list[0], scaling_parameter = True, hyper_priors = HYPER_PRIORS_TOY)
        kernels = np.array(polynomial_kernel_expansion(base_kernels, 2, scaling_parameter = True)) # product kernels (up to order 2) 
        hyper_priors = HYPER_PRIORS_TOY
        p = TOY_INCLUSION_PROB
    else: # real data
        hyper_priors = HYPER_PRIORS_SCALED
        kernels = KERNEL_POOL[data](X_all, trainable = True, rescale = False, hyper_priors = hyper_priors,  interaction = interaction)
        p = np.array([component_inclusion_probability for component in kernels])        
        kernel_product = np.array([isinstance(kernel, gpf.kernels.Product) for kernel in kernels])
        p[kernel_product] = component_inclusion_probability/5.
    print(p)                                                                            
    likelihood_params = {'heartsteps':heartsteps, 'mean_function':gpf.mean_functions.Zero()}
    n_dimensions = X_list[0].shape[1]
    base_dist =  MultinomialKernels(components = kernels, p = p, hyper_priors = hyper_priors, n_dimensions = n_dimensions)
    
    model = MHSelection(X_list[0], y_list[0], base_dist, likelihood = kernel_likelihood, likelihood_params = likelihood_params, seed = train_seed)
    for t in range(len(X_list)):
        print("Running MH for timestep {} of {}".format(t+1, len(X_list)))
        # start at model from previous time step
        if t > 0:
            model.update_data(X_list[t], y_list[t])
        
        # Optimize
        start_time = time.time()
        samples = model.sample(n_iters, burnin, hyper_proposal_variance = hyper_proposal_variance)
        end_time = time.time()
        runtime = end_time - start_time
        
        # Pickle files
        print("RUNTIME: ", runtime)
        results = {"model":model, "runtime":runtime}
        with open("{}chunk_{}.pickle".format(outdir, t), 'wb') as handle:
            pickle.dump(results, handle)


def cks(X_list, y_list, 
        depth = 10, 
        num_restarts = 5, 
        outdir = "", 
        data = "", 
        prior = False, 
        train_seed = 0):

    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)

    heartsteps = False
    if data == "toy":
        if prior: 
            base_kernels = initialize_base_kernels(X_list[0], hyper_priors = None, scaling_parameter = True, rescale_linear = False) 
        else: 
            base_kernels = initialize_base_kernels(X_list[0], hyper_priors = HYPER_PRIORS_TOY, scaling_parameter = True, rescale_linear = False) 
    elif data == "heartsteps": 
        heartsteps = True
        base_kernels = create_hs_kernel_pool(X_list[0], trainable = True, rescale = False, hyper_priors = HYPER_PRIORS_SCALED) 
    
    cks = CompositionalKernelSearch(to_default_float(X_list[0]), 
            to_default_float(y_list[0]),
            seed = train_seed,
            random_restarts = num_restarts, 
            heartsteps = heartsteps, 
            base_kernels = base_kernels,
            priors = prior)

    for t in range(len(X_list)): 
        print("Running CKS for timestep {} of {}".format(t+1, len(X_list)))
        # Update data
        if t > 0: 
            cks.update_data(to_default_float(X_list[t]), to_default_float(y_list[t]))

        # Optimize
        start_time = time.time()
        composition = cks.run(depth)
        end_time = time.time()
        runtime = end_time - start_time
        
        # Pickle files
        print("RUNTIME: ", runtime)
        results = {"model":cks, "runtime":runtime}
        with open("{}chunk_{}.pickle".format(outdir, t), 'wb') as handle:
            pickle.dump(results, handle)

def shrinkage(X_list, y_list,
        data = "toy",
        train_seed = 0,
        outdir = "",
        max_iters = 10000, learning_rate = 0.1, minibatch_proportion = 0.5, scale_parameter = 1e-2, heartsteps = False): 
    
    # random seed
    np.random.seed(train_seed)
    tf.random.set_seed(train_seed)
    
    # Get kernels
    heartsteps = False
    if data == "toy": 
        base_kernels = initialize_base_kernels(X_list[0], scaling_parameter = False, hyper_priors = HYPER_PRIORS_TOY) 
    elif data == "heartsteps": 
        heartsteps = True
        base_kernels = create_hs_kernel_pool(X_list[0], trainable = False, rescale = True, hyper_priors = HYPER_PRIORS_SCALED) 

    # Optimizer params, get minibatch iterator
    optimizer_params = {
        'max_iters':max_iters, 
        'learning_rate':learning_rate,
        'minibatch_proportion':minibatch_proportion, 
    }

    # Initialize shrinkage model
    mean_function = gpf.mean_functions.Zero()
    horseshoe = ShrinkageModel(to_default_float(X_list[0]), 
            to_default_float(y_list[0]), 
            base_kernels = base_kernels, 
            A =scale_parameter, 
            B = 1., 
            mean_function = mean_function, 
            seed = train_seed, 
            heartsteps = heartsteps)
    # Random variance 
    variance_dist =  tfd.LogNormal(to_default_float(0), to_default_float(0.5))
    horseshoe.model.likelihood.variance.prior = variance_dist
    noise_variance = variance_dist.sample()
    horseshoe.model.likelihood.variance.assign(noise_variance)
    print("number of possible kernels: ", len(horseshoe.kernels))

    for t in range(len(X_list)):
        print("Running shrinkage for timestep {} of {}".format(t+1, len(X_list)))
        
        # Update data
        if t > 0:
            horseshoe.model.update_data(to_default_float(X_list[0]), to_default_float(y_list[t]))
    
        # Optimize
        start_time = time.time()
        optimizer_output = horseshoe.optimize(optimizer_params)
        end_time = time.time()
        print(horseshoe.model.selector.map_weights())
        runtime = end_time - start_time
        
        # Pickle files
        print("RUNTIME: ", runtime)
        results = {"model":horseshoe.model,"optimizer_output":optimizer_output, "runtime":runtime}
        with open("{}chunk_{}.pickle".format(outdir, t), 'wb') as handle:
            pickle.dump(results, handle)

