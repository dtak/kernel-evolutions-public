import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow as gpf
from gpflow.utilities import print_summary, set_trainable, to_default_float, deepcopy, positive
from itertools import groupby
from operator import itemgetter
from gpflow import Parameter
import copy
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from .tools.utils import get_data_stats
from .tools.kernels import create_linear, create_period, create_rbf, Interaction, Linear, Periodic
from .DirichletProcess import MarginalGibbsSampler
from .KernelSelection import BIC, MAP

class StratifiedModel: 
    def __init__(self, 
            X_list,                     # dictionary indexed by (m, t) of the data matrices 
            y_list,                     # dictionary indexed by (m, t) of the prediction targets
            z_init = np.ones(0),              # initial clustering of customers. Array whose length is the total number of customers. 
                                        # indexing corresponds to the reservations. For example, if customer (0, 0) is in index
                                        # 0 in restaurant "", then z_init[0] is that customer's table. 
            model_to_string = None,     # function which returns string representation of model structure
            likelihood_func = None,     # function which returns log p(y | X, M, hypers)
            likelihood_params = {},     # parameters of likelihood calculation, contains info on heartsteps and mean function
            hyper_priors = {},          # dictionary mapping from hyperparater element to a TFD distribution. Structure 
                                        # depends on the application. For kernels, hyper_priors['lengthscale'] = LogNormal
            base_distribution_constructor = None, # Constructor for distribution over model structure
            base_distribution_args = {}, # Dictionary of args for base distribution
            seed = 0,
            alpha = 1,                  # DP parameter 
            heartsteps = False,  # whether or not we are predicting for heartsteps (relevant for kernel)
            parent_kernel_prob = None): # bias the children toward the parents kernel.  
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # model parameters
        self.base_distribution_constructor = base_distribution_constructor
        self.base_distribution_args = base_distribution_args
        self.base_distribution_args['hyper_priors'] = hyper_priors
        self.alpha = alpha
        self.to_string = model_to_string
        self.lml = likelihood_func
        self.lml_params = likelihood_params
        self.hyper_priors = hyper_priors # TODO: Is this needed? can we replace by passing list of kernels?

        # Data stats
        self.M, self.T = get_data_stats(X_list)
        self.num_customers = sum(self.T)
        self.T_max = np.max(self.T)
        
        # there is a restaurant per timestep
        self.restaurants = {}
        self.reservations = {} # a dictionary of customers at each restaurant (self.reservations[restaurant] = customers). Indices correspond to the z list. 
        for t in range(self.T_max):
            customers = [(m, t) for m in range(self.M) if self.T[m] > t] # data sets who have at least this many time steps
            self.reservations[t] = customers

            # sample random parameters for each table
            base_distribution = base_distribution_constructor(**self.base_distribution_args)
            z_init = np.zeros(len(customers))
            thetas = [base_distribution.sample() for table in range(np.unique(z_init).shape[0])]
            self.restaurants[t] = MarginalGibbsSampler(alpha, base_distribution, '', z = z_init, thetas = thetas)

        
        self.K = {} # track each customers composition (K[(m, t)] = kernel_name)
        for m in range(self.M): 
            for t in range(self.T[m]): 
                self.K[(m, t)] = "" # TODO: should this be set to the value sampled as the initially sampled thetas? 
        self.kernels = {} # track the kernel itself (self.kernels[(m, t)] = kernel)
        
        # Data
        self.X_list = X_list 
        self.y_list = y_list

        self.heartsteps = heartsteps
        self.parent_kernel_prob = parent_kernel_prob

    def seat_customers(self):
        '''
        assigns customers to tables
        '''
        for t in range(self.T_max):
            dp = self.restaurants[t]
            customers = self.reservations[t]
            if len(customers) > 1: # There is more than one person at this timestep
                for i, mt in enumerate(customers): 
                    # get data
                    X = self.X_list[mt]  
                    y = self.y_list[mt]
                    
                    # Unseat
                    del self.reservations[t][i]
                    dp.unseat(i)

                    # Reseat
                    dp.assign_table(X, y, self.lml, self.lml_params)
                    self.reservations[t].append(mt)
                    
                    # updates
                    new_plate = self.to_string(dp.thetas[int(dp.z[-1])]['structure'])
                    self.K[mt] = new_plate # update composition
                    self.kernels[mt] = dp.thetas[int(dp.z[-1])]
            else:
                new_plate = self.to_string(dp.thetas[int(dp.z[-1])]['structure'])
                self.K[customers[0]] = new_plate # update composition
                self.kernels[customers[0]] = dp.thetas[int(dp.z[-1])]

    def plate_tables(self, mh_params = {}):
        '''
        updates the composition at each table based on who is sitting there
        MH-sampler
        '''
        for t in range(self.T_max):
            dp = self.restaurants[t]
            customers = self.reservations[t]
            dp.update_tables_mh(self.X_list, self.y_list, self.lml, self.lml_params, customers, **mh_params)

            # update K_{m, t} based on what is at table
            for i, customer in enumerate(customers): 
                #self.K[(customer[0], customer[1])] = self.to_string(dp.thetas[int(dp.z[i])])
                self.K[(customer[0], customer[1])] = self.to_string(dp.thetas[int(dp.z[i])]['structure'])
                self.kernels[(customer[0], customer[1])] = dp.thetas[int(dp.z[i])]

    def posterior_likelihood(self):
        ll = 0
        # User likelihood
        user_lmls = self.total_lml()
        
        # Model prior
        structure_prior = 0
        for restaurant, dp in self.restaurants.items(): 
            for table in dp.thetas:
                table_prior = dp.base_dist.log_prob(table)
                structure_prior += table_prior
        print("likelihood: ", user_lmls.numpy(), "prior: ", structure_prior.numpy())
        return float(user_lmls + structure_prior)

    def total_lml(self): 
        lml = 0
        for m in range(self.M): 
            for t in range(self.T[m]):
                new_lml = self.lml(self.X_list[(m, t)], self.y_list[(m, t)], self.kernels[(m, t)], **self.lml_params)
                lml += new_lml
        return lml

    def print(self):
        for m in range(self.M):
            traj = ""
            for t in range(self.T[m]):
                traj += "--> {}/{:.2f}".format(self.K[(m, t)], self.kernels[(m,t)]['noise'])
            print("Task {}: {}".format(m, traj))

    def iterate(self, n_seating = 5, mh_params = {}):
        '''
        goes through one Gibbs sweep of all steps in the following order: seat, plate, update hypers
        '''
        # seat customers
        print("SEATING CUSTOMERS")
        for i in range(n_seating):
            print("Seating iteration {}".format(i))
            self.seat_customers()
        self.print()

        # plate tables
        print("PLATING TABLES")
        self.plate_tables(mh_params)
        self.print()
    
    def select_model(self, X, y, t, learn_hypers = False, validation = None):
        lmls = []
        models = []
        dp = self.restaurants[t]
        for theta in dp.thetas: 
            if learn_hypers: 
                if validation is not None: 
                    X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                    opt_model, bic = BIC(X_train, y_train, theta, dp.base_dist, **self.lml_params)
                    bic = validation(opt_model, X_train, y_train, X_val, y_val)
                else: 
                    opt_model, bic = BIC(X, y, theta, dp.base_dist, **self.lml_params)
            else: 
                if validation is not None: 
                    X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                    opt_model = theta
                    bic = validation(opt_model, X_train, y_train, X_val, y_val)
                else: 
                    opt_model = theta
                    bic = self.lml(X, y, theta, **self.lml_params) + tf.squeeze(dp.base_dist.log_prob(theta)).numpy()

            lmls.append(bic)
            models.append(opt_model)
        print("options: {}".format([self.to_string(mod['structure']) for mod in models]))
        print("lmls: {}".format([tf.squeeze(lik).numpy() for lik in lmls]))
        print("chose {}".format(self.to_string(models[np.argmax(lmls)]['structure'])))
        return models[np.argmax(lmls)]  # Return best leaf
