TOY_PARAMS = {
        'evolution': ('toy-evolution', dict(
            n_gibbs_iters = [400], # Method args
            n_mh_iterations = [100],
            n_seating = [10],
            alpha = [1],
            adapt_noise_prior = [True],
            mh_hyper_proposal_variance = [0.1], 
            parent_kernel_prob = [0.9], 
            train_seed = range(3), 
            M = [10], # Data args 
            noise = [0.5], 
            data_seed = range(3),
            ground_kernel = ["simpsamp"],
            num_per_timestep = [[5 for i in range(6)]]
            )), 
        'final': ('toy-final', dict(
                n_gibbs_iters = [400], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.1], 
                adapt_noise_prior = [False],
                train_seed = range(3), 
                M = [10], # Data args 
                noise = [0.5], 
                data_seed = range(3),
                ground_kernel = ["complex"],
                num_per_timestep = [[5 for i in range(6)]]
                )),
        'stratified': ('toy-stratified', dict(
                n_gibbs_iters = [400], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                adapt_noise_prior = [False],
                mh_hyper_proposal_variance = [0.1], 
                train_seed = range(3), 
                M = [10], # Data args 
                noise = [0.5], 
                data_seed = range(3),
                ground_kernel = ["sim1"],
                num_per_timestep = [[5 for i in range(6)]]
                )), 
        'mh': ('toy-mh', dict(
                n_iters = [10000], # Method args
                hyper_proposal_variance = [0.1], 
                train_seed = [2], 
                M = [50], # Data args 
                noise = [0.5], 
                data_seed = [1000],
                user = [6, 16],
                adapt_noise_prior = [False],
                ground_kernel = ["sim1"],
                num_per_timestep = [[5 for i in range(6)]]
                )),
        'ard': 
            ('toy-ard', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.1], 
                prior = [True],
                train_seed = range(3),
                M = [50], # Data args 
                user = range(50),
                adapt_noise_prior = [False],
                noise = [0.5], 
                data_seed = [1000],
                ground_kernel = ["sim1"],
                num_per_timestep = [[5 for i in range(6)]]
                ))
        }
