'''
Implements various forms/approximations of the Dirichlet Process
'''

import numpy as np
import tensorflow as tf
import scipy as sp
import gpflow as gpf
from scipy.special import logsumexp
from gpflow.utilities import print_summary
class MHSampler: 
    def __init__(self, base_dist, p_add = 0.2, p_stay = 0.4, p_remove = 0.4):
        self.base_dist = base_dist
         
        if p_add + p_remove + p_stay != 1: 
            raise ValueError("MH-sampler: Probability of adding and removing must sum to one!")
        self.p_add = p_add
        self.p_stay = p_stay
        self.p_remove = p_remove
        self.num_options = self.base_dist.n

    def sample(self, structure, hypers, target, hyper_proposal_variance = 1): 
        '''
        returns model after one iteration of sampling, along with a 1 if accepted, 0 if not
        '''
        # propose a model
        proposal_structure, log_prop_ratio = self.propose_discrete_structure(structure) # propose a new model structure
        proposal_hypers = self.propose_hypers(hypers, hyper_proposal_variance = hyper_proposal_variance) # propose new model hypers

        # calculate acceptance ratio
        U = np.random.uniform(0, 1)
        log_current = target(structure, hypers)
        log_proposal = target(proposal_structure, proposal_hypers)
        log_prob_ratio = log_proposal - log_current
        ratio = np.exp(log_prob_ratio + log_prop_ratio)
        if U < np.min((1, ratio)): # accept proposal
            #print("current {}, proposed {}".format("+".join(self.base_dist.kernel_names[structure > 0]), "+".join(self.base_dist.kernel_names[proposal_structure > 0])))
            #print("ACCEPTED\n")
            #print_summary(self.base_dist.to_model(proposal_structure, proposal_hypers)['structure'])
            return proposal_structure, proposal_hypers, 1 
        else:
            return structure, hypers, 0

    def propose_hypers(self, hypers, hyper_proposal_variance = 1.):
        # add random normal
        return np.copy(hypers) + np.random.normal(0, hyper_proposal_variance, size = hypers.shape)

    
    def find_structure_state(self, num_included):
        if num_included > 0 and num_included < self.num_options:  # intermediate number
            action = np.random.choice(["remove", "add", "stay"], p = [self.p_remove, self.p_add, self.p_stay])
            curr_state = 0
        elif num_included >= self.num_options: # max number of kernels
            action = np.random.choice(["remove","stay"])
            curr_state = 1
        else:
            action = np.random.choice(["add", "stay"])
            curr_state = 2

        return curr_state, action 
    
    def discrete_proposal_prob(self, state, action, num_included):
        #print(num_included, " ", self.num_options)
        if state == 0: # intermediate
            if action == "add":
                return self.p_add * 1./(self.num_options - num_included),
            elif action == "remove":
                return self.p_remove * 1./num_included
            else: 
                return self.p_stay
        elif state == 1: # full 
            if action == "add":
                return 0
            elif action == "remove":
                return 0.5 * 1./num_included
            else: 
                return 0.5
        else: # under
            if action == "add":
                return 0.5 * 1./(self.num_options - num_included)
            elif action == "remove":
                return 0
            else: 
                return 0.5

    def propose_discrete_structure(self, vec):
        # model: binary inclusion vector for kernels/features
        # TODO: check this logic
        # Nonsymettric
        # randomly add or remove a kernel

        num_included = np.count_nonzero(vec)
        num_options = self.base_dist.n
        proposal = np.copy(vec) # TODO: Do I need tihs?
       
        curr_state, action = self.find_structure_state(num_included)

        # apply action and find reverse action
        if action == "add":
            zeros = np.where(vec == 0)[0]
            proposal[np.random.choice(zeros)] = 1
            reverse_action = "remove"
        elif action == "remove":
            ones = np.where(vec == 1)[0]
            proposal[np.random.choice(ones)] = 0
            reverse_action = "add"
        else: 
            reverse_action = "stay"

        # find next state
        proposal_num_included = np.count_nonzero(proposal)
        proposal_state, _ = self.find_structure_state(proposal_num_included)
        
        # calculate probabilities
        p_x_y = self.discrete_proposal_prob(curr_state, action, num_included) # p(proposal | current)
        p_y_x = self.discrete_proposal_prob(proposal_state, reverse_action, proposal_num_included) # p(current | proposal)

        return proposal, np.log(p_x_y) - np.log(p_y_x)


class MarginalGibbsSampler: 
    '''
    Collapsed pi, initializing large number of clusters C
    '''
    def __init__(self, alpha, base_dist, parent, z = np.ones(0), thetas = None): 
        '''
        alpha: Concentration parameter for DP
        C: number of tables
        '''
        self.alpha = alpha # concentration parameter
        self.base_dist = base_dist # H_0
        self.z = z.astype(int)
        if thetas is None: 
            self.thetas = []
        else: 
            self.thetas = thetas

        self.num_customers = self.z.shape[0]
        self.num_tables = np.unique(self.z).shape[0]
        self.parent = parent
        self.sampler = MHSampler(base_dist)
#        print(self.base_dist.hyper_priors)
    
    def unseat(self, i):
        '''
        Unseats the i-th customer from the table
        '''
        table = int(self.z[i]) # see which table the customer was at
        self.num_customers -= 1
        self.z = np.delete(self.z, i)
        new_num_tables = np.unique(self.z).shape[0]
        
        # relabel if necessary, customer was only one at table
        if new_num_tables != self.num_tables: 
            unq_arr, self.z = np.unique(self.z ,return_inverse=1)
            del self.thetas[table] # remove table, now empty
        self.num_tables = new_num_tables

        # TODO check this carefully


    def assign_table(self, X, y, log_likelihood_function, likelihood_function_params, prev_dish = None):
        '''
        Assigning table for the n-th customer. 

        X: data for the n-th customer
        y: target values for the n-th customers data
        log_likelihood_function: 
        '''
        # CRP table assignment probabilities
        if self.num_customers > 0: 
            log_denom = np.log(self.num_customers - 1. + self.alpha)
            tables, counts = np.unique(self.z, return_counts = True)
            existing_probs = np.log(counts) - log_denom
            new_prob = np.log(self.alpha) - log_denom # multinomial probability for new table
            crp_probs = np.append(existing_probs, new_prob)
            
            # Get customer's likelihood of sitting at each table
            likelihoods = np.array([log_likelihood_function(X, y, theta, **likelihood_function_params ) for theta in self.thetas])
            new_params = self.base_dist.sample() # option to start a new table, with parameters sampled from H_0
            likelihoods = np.append(likelihoods, log_likelihood_function(X, y, new_params, **likelihood_function_params))
            # assign a table
            final_probs = crp_probs + likelihoods
            final_denom = logsumexp(final_probs)
            multinomial_probs = np.exp(final_probs - final_denom) # raise back

            if np.sum(multinomial_probs) == 0:
                print("ZEROed out")
                num = len(multinomial_probs)
                multinomial_probs = np.ones(num) * 1./num
            else: 
                multinomial_probs = multinomial_probs / np.sum(multinomial_probs)
            table = np.random.choice(self.num_tables + 1, p = multinomial_probs)
        else: # opening a new restaurant
            table = 0
            new_params = prev_dish # if opening a new restaurant, set dish as the old dish
            
        
        # Seat the customer
        self.z = np.append(self.z, int(table))
        self.num_customers +=1 
        if table == self.num_tables: # new table was created
            self.thetas.append(new_params)
            self.num_tables +=1
            '''
            print("Accepted new table")
            for i, theta in enumerate(self.thetas): 
                likelihood = log_likelihood_function(X, y, theta, **likelihood_function_params)
                print("table {}:\nstructure: {}\nlikelihood: {}".format(i, self.base_dist.to_string(theta), likelihood))
                print_summary(theta['structure'])
            print("\n\n") 
                '''
        else: 
            del new_params
#        new = table == self.num_tables
        
#        return table, new, new_params # TODO: Check returns
    
    def print(self):
        print("Num customers: {}".format(self.num_customers))
        print("Seating: {}".format(self.z))
        print("Dishes")
        for i, theta in enumerate(self.thetas): 
            print("Table {}: {}".format(i, self.base_dist.to_string(theta)))
        

    def table_likelihood(self, X_list, y_list, customers, func, func_params, kernel):
        likelihoods = [func(X_list[mt], y_list[mt], kernel, **func_params) for mt in customers]
        return np.sum(likelihoods)
    
    def update_tables_mh_full(self, X_list, y_list, log_likelihood_function, likelihood_function_params, customers, num_iterations = 100, burnin = 0.4, hyper_proposal_variance = 1.):
        '''
        sample from posterior at each table
        customers: array of [(m, t)] where each index corresponds to the z index
        '''
        for table, table_model in enumerate(self.thetas):
            print("Updating table {} of {}: {}".format(table + 1, self.num_tables, self.base_dist.to_string(table_model)))
            table_customers = [(mt[0], mt[1]) for i, mt in enumerate(customers) if self.z[i] == table]
            
            # define the posterior
            def target(structure, hypers): 
                model = self.base_dist.to_model(structure, hypers)
                #print_summary(model)
                likelihood = self.table_likelihood(X_list, y_list, table_customers, log_likelihood_function, likelihood_function_params, model)
                prior = self.base_dist.log_prob(model)
                del model
                #print("likelihood {}, prior {}".format(likelihood, prior))
                return likelihood + prior # posterior
            
            # MH samples
            structure_current = self.base_dist.get_structure(table_model) # start
            hypers_current = self.base_dist.get_full_hypers(table_model)
            samples = []
            accepts = 0
            for i in range(num_iterations):
#                print("curr model {}".format(self.base_dist.kernel_names[structure_current > 0]))
                sample_struct, sample_hypers, accepted = self.sampler.sample(structure_current, hypers_current, target,
                        hyper_proposal_variance = hyper_proposal_variance) # generate MH sample
                hyper_diff = np.linalg.norm(sample_hypers - hypers_current)
#                if accepted > 0: 
#                    print("ACCEPTED! model {} and hypers {}".format(self.base_dist.kernel_names[sample_struct > 0], hyper_diff))
                accepts += accepted
                samples.append(self.base_dist.to_model(sample_struct, sample_hypers, copy = True))
                structure_current = sample_struct
                hypers_current = sample_hypers

                if i % 50 == 0: 
                    print("MH iteration {}".format(i))

            # Burn in 
            samples = samples[int(accepts * burnin):]
            print("Num samples {}, Accept proportion: {}".format(len(samples), accepts * 1./num_iterations))
            if len(samples) > 0: 
                sample = samples[np.random.choice(len(samples))]
                self.thetas[table] = sample 
            
            print("CHOICE: ", self.base_dist.to_string(self.thetas[table]))

            # Cleanup
            del target
            for sample in samples: 
                del sample
            del samples

    def update_tables_mh(self, X_list, y_list, log_likelihood_function, likelihood_function_params, customers, num_iterations = 100, burnin = 0.4, hyper_proposal_variance = 1.):
        '''
        sample from posterior at each table
        customers: array of [(m, t)] where each index corresponds to the z index
        '''
        for table, table_model in enumerate(self.thetas):
            print("Updating table {} of {}: {}".format(table + 1, self.num_tables, self.base_dist.to_string(table_model)))
            max_customers = {}
            for i, mt in enumerate(customers):
                if self.z[i] == table: 
                    m = mt[0]
                    t = mt[1]
                    if m in max_customers: 
                        t_curr = max_customers[m]
                        if t > t_curr: 
                            max_customers[m] = t
                    else: 
                        max_customers[m] = t
#            table_customers = [(mt[0], mt[1]) for i, mt in enumerate(customers) if self.z[i] == table]
            table_customers = [(m, t) for m, t in max_customers.items()]
            
            # define the posterior
            def target(structure, hypers): 
                model = self.base_dist.to_model(structure, hypers)
                #print_summary(model)
                likelihood = self.table_likelihood(X_list, y_list, table_customers, log_likelihood_function, likelihood_function_params, model)
                prior = self.base_dist.log_prob(model)
                del model
                #print("likelihood {}, prior {}".format(likelihood, prior))
                return likelihood + prior # posterior
            
            # MH samples
            structure_current = self.base_dist.get_structure(table_model) # start
            hypers_current = self.base_dist.get_full_hypers(table_model)
            samples = []
            accepts = 0
            for i in range(num_iterations):
#                print("curr model {}".format(self.base_dist.kernel_names[structure_current > 0]))
                sample_struct, sample_hypers, accepted = self.sampler.sample(structure_current, hypers_current, target,
                        hyper_proposal_variance = hyper_proposal_variance) # generate MH sample
                hyper_diff = np.linalg.norm(sample_hypers - hypers_current)
#                if accepted > 0: 
#                    print("ACCEPTED! model {} and hypers {}".format(self.base_dist.kernel_names[sample_struct > 0], hyper_diff))
                accepts += accepted
                samples.append(self.base_dist.to_model(sample_struct, sample_hypers, copy = True))
                structure_current = sample_struct
                hypers_current = sample_hypers

                if i % 50 == 0: 
                    print("MH iteration {}".format(i))

            # Burn in 
            samples = samples[int(accepts * burnin):]
            print("Num samples {}, Accept proportion: {}".format(len(samples), accepts * 1./num_iterations))
            if len(samples) > 0: 
                sample = samples[np.random.choice(len(samples))]
                self.thetas[table] = sample 
            
            print("CHOICE: ", self.base_dist.to_string(self.thetas[table]))

            # Cleanup
            del target
            for sample in samples: 
                del sample
            del samples
    
