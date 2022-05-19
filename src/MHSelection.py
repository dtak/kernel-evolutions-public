from .DirichletProcess import MHSampler
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary
class MHSelection:
    def __init__(self, X, y, base_dist, likelihood = None, likelihood_params = None, seed = 0): 
        self.X = X
        self.y = y
        self.base_dist = base_dist
        self.likelihood = likelihood
        self.likelihood_params = likelihood_params
        
        self.model = self.base_dist.sample()
        print_summary(self.model)
        print(self.model)
        self.sampler = MHSampler(base_dist)
        
        # filled from sampling
        self.samples = []
        self.posterior_likelihoods = []

        np.random.seed(seed)
        tf.random.set_seed(seed)
    def target(self, structure, hypers):
        model = self.base_dist.to_model(structure, hypers)
        likelihood = self.likelihood(self.X, self.y, model, **self.likelihood_params)
        prior = self.base_dist.log_prob(model)
        del model
        return likelihood + prior

    def update_data(self, X, y):
        self.X = X
        self.y = y

    def sample(self, n_iters, burnin, hyper_proposal_variance = 0.1): 
        # MH samples
        structure_current = self.base_dist.get_structure(self.model) # start
        hypers_current = self.base_dist.get_full_hypers(self.model)

        samples = []
        posterior_likelihoods = []
        accepts = 0
        for i in range(n_iters):
            sample_struct, sample_hypers, accepted = self.sampler.sample(structure_current, hypers_current, self.target, 
                    hyper_proposal_variance = hyper_proposal_variance) # generate MH sample
            accepts += accepted
            if accepted == 1 or len(posterior_likelihoods) == 0: 
                posterior_likelihoods.append(self.target(sample_struct, sample_hypers)) # recalculate
            else: 
                posterior_likelihoods.append(posterior_likelihoods[-1]) # same as before

            samples.append(self.base_dist.to_model(sample_struct, sample_hypers, copy = True))
            structure_current = sample_struct
            hypers_current = sample_hypers
            
            if i % 50 == 0: 
                target = self.target(structure_current, hypers_current)
                print("MH iteration {}, current structure {}, target {}".format(i, self.base_dist.to_string(samples[-1]), target))

        # Burn in 
        burned_index = int(accepts * burnin)
        burned_samples = samples[burned_index:]
        burned_likelihoods = posterior_likelihoods[burned_index:]
        print("Num samples {}, Accept proportion: {}".format(len(samples), accepts * 1./n_iters))
        # set new sample
        if len(samples) > 0: 
            sample = burned_samples[np.argmax(burned_likelihoods)] # MAP SAMPLE
            self.model = sample
        print("CHOICE: {}, {}".format(self.base_dist.to_string(sample), np.max(burned_likelihoods)))
        
        last_index = 1000 if len(samples) > 1000 else len(samples)
        self.samples = np.array(samples[-last_index:]) # only store last 1000
        self.posterior_likelihoods = np.array(posterior_likelihoods[-last_index:])

        return samples
        
    def total_lml(self): 
        return self.likelihood(X, y, self.model, **self.likelihood_params)
