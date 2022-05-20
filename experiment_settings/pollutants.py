    ####################################
    ####### POLLUTANTS          ########
    ####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/cluster"
        method_name = "cluster_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/traj-full"
        method_name = "traj_kernel_selection_pollutants_full"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                interaction = [True],
                adapt_noise_prior = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/traj"
        method_name = "traj_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                interaction = [True],
                adapt_noise_prior = [True],
                train_seed = [1], 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = [0]
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "pollutants":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/stratified"
        method_name = "stratified_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [12], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "pollutants": 
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/pollutants/ard"
        method_name = "ard_kernel_selection_pollutants"
        QUEUE.append(
            ('edit-uci', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [24], # Data args 
                chunk_size = [5],
                user = range(24),
                data_seed = [0]
                )
            ))
