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

class TrajectoryModel: 
    def __init__(self, 
            X_list,                     # dictionary indexed by (m, t) of the data matrices 
            y_list,                     # dictionary indexed by (m, t) of the prediction targets
            z_init = np.ones(0),              # initial clustering of customers. Array whose length is the total number of customers. 
                                        # indexing corresponds to the reservations. For example, if customer (0, 0) is in index
                                        # 0 in restaurant "", then z_init[0] is that customer's table. 
            reservations = None,        # dictionary which maps restaurant (string representation of model) to a list 
                                        # of customers (reservations[""] = [(0, 0), (0, 1), (1, 0), (1, 1)])
            model_to_string = None,     # function which returns string representation of model structure
            likelihood_func = None,     # function which returns log p(y | X, M, hypers)
            likelihood_params = {},     # parameters of likelihood calculation, contains info on heartsteps and mean function
            hyper_priors = {},          # dictionary mapping from hyperparater element to a TFD distribution. Structure 
                                        # depends on the application. For kernels, hyper_priors['lengthscale'] = LogNormal
#            hyper_constructor = None,   # Hyperparameter class constructor. Depends on application. 
            base_distribution_constructor = None, # Constructor for distribution over model structure
            base_distribution_args = {}, # Dictionary of args for base distribution
            seed = 0,
            alpha = 1,                  # DP parameter 
            heartsteps = False,  # whether or not we are predicting for heartsteps (relevant for kernel)
            adapt_noise_prior = False,  # whether or not to adapt noise prior for children
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
        # hyperparameters
        self.hyper_priors = hyper_priors # TODO: Is this needed? can we replace by passing list of kernels?
        self.adapt_noise_prior = adapt_noise_prior
        
        # GLOBAL paramaters (things indexed by restaurant)
        self.restaurants = {}
        # sample random parameters for each table
        base_distribution = base_distribution_constructor(**self.base_distribution_args)
        if adapt_noise_prior: 
            base_distribution.hyper_priors['noise'] = self.noise_prior("") # parent has 0 elements
        thetas = [base_distribution.sample() for table in range(np.unique(z_init).shape[0])]
        self.restaurants[''] = MarginalGibbsSampler(alpha, base_distribution, '', z = z_init, thetas = thetas)

        # LOCAL parameters (things indexed by m, t)
        self.M, self.T = get_data_stats(X_list)
        self.num_customers = sum(self.T)

        self.K = {} # track each customers composition (K[(m, t)] = kernel_name)
        for m in range(self.M): 
            for t in range(self.T[m]): 
                self.K[(m, t)] = "" # TODO: should this be set to the value sampled as the initially sampled thetas? 

        self.kernels = {} # track the kernel itself (self.kernels[(m, t)] = kernel)
        self.reservations = reservations # a dictionary of customers at each restaurant (self.reservations[restaurant] = customers). Indices correspond to the z list. 
        
        # Data
        self.X_list = X_list 
        self.y_list = y_list

        self.heartsteps = heartsteps
        self.parent_kernel_prob = parent_kernel_prob

    def seat_customers(self):
        '''
        assigns customers to tables
        '''
        for m in range(self.M): # for each task
            #print("Seating customer {}".format(m))
            for t in range(self.T[m] -1, -1, -1): # for each time point, in reverse TODO: check if needs + 1
            #for t in range(self.T[m]): # for each time point TODO: check if needs + 1
                # get data
                X = self.X_list[(m, t)]  
                y = self.y_list[(m, t)]
                # get actual restaurant, index of customer, unseat
                for restaurant, customers in self.reservations.items(): 
                    if (m, t) in customers: # found customer 
                        # note location
                        og_restaurant = restaurant
                        i = customers.index((m, t))
                        break
                # unseat customer
                del self.reservations[og_restaurant][i]
                dp = self.restaurants[og_restaurant]
                prev_dish = copy.deepcopy(dp.thetas[dp.z[i]]) # track the model at the table the customer used to sit at
                dp.unseat(i)
                
                # close restaurant if empty
                if dp.num_customers == 0: 
                    del self.restaurants[og_restaurant] # remove
                    del self.reservations[og_restaurant]
                
                # seat user in updated restaurant
                if t > 0: 
                    restaurant = self.K[(m, t - 1)]
                else: 
                    restaurant = ""
                
                if restaurant not in self.restaurants:
                    #print("ADDING")
                    #print(self.parent_kernel_prob, " ", t)
                    H0 = self.base_distribution_constructor(**self.base_distribution_args)
                    # probability towards parent kernel (restaurant)
                    if self.parent_kernel_prob is not None and t > 0: 
                        kernel_vec = H0.get_structure(self.kernels[(m, t-1)])
                        H0.p[kernel_vec > 0] = self.parent_kernel_prob 
                    
                    # noise prior
                    if self.adapt_noise_prior: 
                        H0.hyper_priors['noise'] = self.noise_prior(restaurant) # parent has 0 elements
                    #    print("updated inclusion")
                    self.restaurants[restaurant] = MarginalGibbsSampler(self.alpha, H0, restaurant) # new dp
                
                dp = self.restaurants[restaurant]
                
                # assign to a table at restaurant 
                dp.assign_table(X, y, self.lml, self.lml_params,  prev_dish = prev_dish)

                # is now the n-th customer at the restaurant, so update the location!
                if restaurant in self.reservations:
                    self.reservations[restaurant].append((m, t))
                else: 
                    self.reservations[restaurant] = [(m, t)]

                # updates
                og_plate = self.K[(m, t)]
                #new_plate = self.to_string(dp.thetas[int(dp.z[-1])])
                new_plate = self.to_string(dp.thetas[int(dp.z[-1])]['structure'])
                self.K[(m, t)] = new_plate # update composition
                self.kernels[(m, t)] = dp.thetas[int(dp.z[-1])]

        if np.sum([len(guests) for rest, guests in self.reservations.items()]) != self.num_customers:
            raise ValueError("Did not pass seating check! Number seating != num customers")
    
    def noise_prior(self, model):
        '''
        determines noise prior based on number of parent's components
        '''
        if model == "" or "White" in model: 
            num_components = 0
        else: 
            num_components = len(model.split("+"))
        #print(model, "has ", num_components, "components")
        if num_components == 0:
            return {'mean':2, 'var': 0.5}
        elif num_components == 1: 
            return {'mean':1, 'var': 1}
        else: 
            return {'mean':0, 'var': 2}


    def plate_tables(self, mh_params = {}):
        '''
        updates the composition at each table based on who is sitting there
        MH-sampler
        '''

        # Make sure restaurants and locations sync
        '''
        location_rest = np.array(list(self.reservations.keys()))
        location_rest_counts = np.array([len(val) for val in self.reservations.values()])
        rest_counts = np.array([dp.num_customers for rest, dp in self.restaurants.items()])
        rest = np.array(list(self.restaurants.keys()))
        sort = np.argsort(rest_counts)
        '''
        for restaurant, dp in self.restaurants.items():
            if dp.num_customers > 0: # non leaf node
                # Get ordered list of customers at the restuarant
                customers = self.reservations[restaurant]
                dp.update_tables_mh_full(self.X_list, self.y_list, self.lml, self.lml_params, customers, **mh_params)

                # update K_{m, t} based on what is at table
                for i, customer in enumerate(customers): 
                    #self.K[(customer[0], customer[1])] = self.to_string(dp.thetas[int(dp.z[i])])
                    self.K[(customer[0], customer[1])] = self.to_string(dp.thetas[int(dp.z[i])]['structure'])
                    self.kernels[(customer[0], customer[1])] = dp.thetas[int(dp.z[i])]

        # Reconstruct structure
        new_reservations = {}
        new_restaurants = {}
        for m in range(self.M): 
            for t in range(self.T[m]): 
                model = self.kernels[(m, t)]
                model_string = self.K[(m, t)]
                if t > 0: 
                    restaurant = self.K[(m, t - 1)]
                else: 
                    restaurant = ""

                if restaurant in new_restaurants: # already started DP
                    dp = new_restaurants[restaurant]
                    dishes = [self.to_string(theta['structure']) for theta in dp.thetas]
                    new_reservations[restaurant].append((m, t))
                    # place in restaurant
                    if model_string in dishes: # dish already exists, place at table
                        index = dishes.index(model_string)
                        dp.z = np.append(dp.z, index)
                        dp.num_customers += 1
                    else: 
                        dp.thetas.append(model) # TODO: not sure if by reference...buggy? 
                        dp.z = np.append(dp.z, dp.num_tables)
                        dp.num_tables += 1
                        dp.num_customers +=1
                else: # start new restaurant, place customer
                    H0 = self.base_distribution_constructor(**self.base_distribution_args)
                    # probability towards parent kernel (restaurant)
                    if self.parent_kernel_prob is not None and t > 0: 
                        kernel_vec = H0.get_structure(self.kernels[(m, t-1)])
                        H0.p[kernel_vec > 0] = self.parent_kernel_prob 
                    
                    # noise prior
                    if self.adapt_noise_prior: 
                        H0.hyper_priors['noise'] = self.noise_prior(restaurant) 
                    
                    new_restaurants[restaurant] = MarginalGibbsSampler(self.alpha, H0, restaurant, thetas = [model], z = np.array([0])) # new dp
                    new_reservations[restaurant] = [(m, t)]
        del self.restaurants
        del self.reservations
        self.restaurants = new_restaurants
        self.reservations = new_reservations

    def posterior_likelihood(self):
        '''
        ll = 0
        dp = self.restaurants[""]
        for m in range(self.M): 
            for t in range(self.T[m]):
                new_likelihood = self.lml(self.X_list[(m, t)], self.y_list[(m, t)], self.kernels[(m, t)], **self.lml_params)
        
        new_prior = tf.squeeze(dp.base_dist.log_prob(self.kernels[(m,t)])).numpy()
        ll += (new_likelihood + new_prior)
        '''
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

    def _form_leaves(self):
        self.leaves = {}
        for m in range(self.M):
            model = self.kernels[(m, self.T[m] - 1)] # Get all the "leaves" (final kernels)
            # Get prior
            if self.T[m] > 1: 
                restaurant = self.K[(m, self.T[m] - 2)]
            else: 
                restaurant = ""
            
            base_dist = self.restaurants[restaurant].base_dist
            self.leaves[restaurant + self.to_string(model['structure'])] = (model, base_dist)
    
    def select_leaf(self, X, y, learn_hypers = False, validation = None):
        lmls = []
        models = []
        for structure, selection in self.leaves.items():
            if learn_hypers: 
                if validation is not None: 
                    X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                    opt_model, bic = BIC(X_train, y_train, selection[0], selection[1], **self.lml_params)
                    bic = validation(opt_model, X_train, y_train, X_val, y_val)
                else: 
                    opt_model, bic = BIC(X, y, selection[0], selection[1], **self.lml_params)
            else: 
                if validation is not None: 
                    X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                    opt_model = selection[0]
                    bic = validation(opt_model, X_train, y_train, X_val, y_val)
                else: 
                    opt_model = selection[0]
                    bic = self.lml(X, y, selection[0], **self.lml_params) + tf.squeeze(selection[1].log_prob(selection[0])).numpy()

            lmls.append(bic)
            models.append(opt_model)
        print("leaf options: {}".format([self.to_string(mod['structure']) for mod in models]))
        print("leaf lmls: {}".format([tf.squeeze(lik).numpy() for lik in lmls]))
        print("chose leaf {}".format(self.to_string(models[np.argmax(lmls)]['structure'])))
        return models[np.argmax(lmls)], lmls # Return best leaf
    
    def form_strata(self):
        T_max = np.max(self.T)
        self.strata = []
        for t in range(T_max):
            # See what other users had here
            options = {}
            for m in range(self.M): 
                option = self.kernels[(m, t)]
                if option['structure'] not in options: 
                    options[option['structure']] = option
            options = list(options.values())
            print("Options at time {}: {}".format(t, [self.to_string(k) for k in options]))
            self.strata.append(options)

    def predict_trajectory(self, X_train_list, y_train_list, threshold = 1, prioritize_leaves = False, learn_hypers = False, validation = None):
        '''
        traverse tree conditionally
        '''
        curr_model_string = ""
        curr_model = None
        curr_model_initial = None
        T = len(y_train_list)
        lmls = []
        models = []
        at_leaf = False
        for t in range(T):
            print("SHAPE", X_train_list[t].shape)
#            print("Curr node: {}".format(curr_model_string))
            # data
            X = X_train_list[t]
            y = y_train_list[t]
            if prioritize_leaves: 
                leaf, likelihoods = self.select_leaf(X, y, learn_hypers = learn_hypers, validation = validation)
                likelihoods = np.sort(likelihoods)
                traverse = False
                if likelihoods.shape[0] >= 2: # At least two options
                    traverse = np.abs(likelihoods[-1] - likelihoods[-2]) < threshold # ambiguity between two options
            else: 
                traverse = True
            if traverse:
                # get current restaurant
                if curr_model_string in self.restaurants: 
                    dp = self.restaurants[curr_model_string]

                if dp.num_tables > 0: # not a leaf node
                    # add self loop 
                    model_represented = (curr_model_string in [self.to_string(theta['structure']) for theta in dp.thetas])
                    #model_represented = False
                    thetas = list(dp.thetas)
                    if model_represented == False and curr_model is not None:
                    #if curr_model is not None: 
                        #thetas.append(curr_model_initial)
                        thetas.append(curr_model)
                    print(thetas)
                    # get table with the highest model likelihood
                    tables = []
                    table_likelihoods = []
                    for theta in thetas:
                        if learn_hypers: 
                            if validation is not None: 
                                X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                                table_model, bic = BIC(X_train, y_train, theta, dp.base_dist, **self.lml_params)
                                bic = validation(table_model, X_train, y_train, X_val, y_val)
                            else: 
                                table_model, bic = BIC(X, y, theta, dp.base_dist, **self.lml_params)
                        else: 
                            table_model = theta
                            if validation is not None: 
                                X_train, X_val, y_train, y_val = train_test_split(X, np.array(y), test_size=0.33)
                                bic = validation(theta, X_train, y_train, X_val, y_val)
                            else: 
                                bic = self.lml(X, y, theta, **self.lml_params) + tf.squeeze(dp.base_dist.log_prob(theta)).numpy()
                        
                        table_likelihoods.append(bic)
                        tables.append(table_model)
    
                    # get the new model
                    top_i= np.argsort(table_likelihoods)
                    next_table = top_i[-1]
                    curr_table = tables[next_table]
                    curr_model_initial = thetas[next_table]
                    curr_model = {key:item for key, item in curr_table.items()}
                    curr_model_string = self.to_string(curr_model['structure'])
                    models.append(curr_model)

                    print("options: {}".format([self.to_string(mod['structure']) for mod in tables]))
                    print("lmls: {}".format([tf.squeeze(lik).numpy() for lik in table_likelihoods]))
                    print("chose: {}".format(curr_model_string))

                else: # leaf node, stay here
                    at_leaf = True
                    curr_model = {key:item for key, item in curr_model.items()}
                    curr_model_string = self.to_string(curr_model['structure'])
                    if learn_hypers: 
                        curr_model, bic = BIC(X, y, curr_model_initial, dp.base_dist, **self.lml_params)
                    print("leaf node {}".format(curr_model_string))
                    models.append(curr_model)
            else: 
                models.append(leaf)
                        
        return models        
