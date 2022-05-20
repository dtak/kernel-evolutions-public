TOY_EXP_PARAMS = {
        'evolution': ,
        'final': ,
        'stratified', 
        'mh': ,
        'ard': 
        }
    ####################################
    ####### CHLORIDES           ########
    ####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/cluster"
        method_name = "cluster_kernel_selection_chlorides"
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
                M = [21], # Data args 
                chunk_size = [5],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/traj-full"
        method_name = "traj_kernel_selection_chlorides_full"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/traj"
        method_name = "traj_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                parent_kernel_prob = [0.9], 
                adapt_noise_prior = [True],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "chlorides":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/stratified"
        method_name = "stratified_kernel_selection_chlorides"
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
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "chlorides": 
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/chlorides/ard"
        method_name = "ard_kernel_selection_chlorides"
        QUEUE.append(
            ('edit-uci', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [31], # Data args 
                chunk_size = [5],
                user = range(31),
                data_seed = range(3)
                )
            ))
    #####################################
    ######## WINE                ########
    #####################################
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj-full" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/traj-full"
        method_name = "traj_kernel_selection_wine_full"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                adapt_noise_prior = [True],
                interaction = [True],
                parent_kernel_prob = [0.9], 
                train_seed = range(3), 
                M = [21], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "traj" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/traj"
        method_name = "traj_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                adapt_noise_prior = [True],
                interaction = [True],
                parent_kernel_prob = [0.9], 
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "cluster" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/cluster"
        method_name = "cluster_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = [0], 
                M = [21], # Data args 
                chunk_size = [5],
                data_seed = [0,1]
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "stratified" and DATA == "wine":
        OUTPUT_DIR = STORAGE_DIR + "icml/kernel-selection/wine/stratified"
        method_name = "stratified_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_gibbs_iters = [200], # Method args
                n_mh_iterations = [100],
                n_seating = [10],
                alpha = [1],
                mh_hyper_proposal_variance = [0.05], 
                component_inclusion_probability = [0.1],
                interaction = [True],
                train_seed = range(3), 
                M = [8], # Data args 
                chunk_size = [50],
                data_seed = range(3)
                )
            ))
    elif EXP_TYPE == "kernel-selection" and METHOD == "mh" and DATA == "wine":
        OUTPUT_DIR = HOME_DIR + "icml/kernel-selection/wine/mh"
        method_name = "mh_kernel_selection_wine"
        QUEUE.append(
            ('stricter', dict(
                n_iters = [10000], # Method args
                hyper_proposal_variance = [0.05],
                component_inclusion_probability = [0.05],
                train_seed = range(3),   
                M = [12], # Data args 
                chunk_size = [50],
                user = range(12),
                data_seed = [0]
            ))
        )
    elif EXP_TYPE == "kernel-selection" and METHOD == "ard" and DATA == "wine": 
        OUTPUT_DIR = HOME_DIR + "icml/kernel-selection/wine/ard"
        method_name = "ard_kernel_selection_wine"
        QUEUE.append(
            ('post-icml', dict(
                n_restarts = [1], 
                hyper_proposal_variance = [0.05], 
                prior = [True],
                train_seed = range(3),
                M = [31], # Data args 
                chunk_size = [5],
                user = range(31),
                data_seed = range(3)
                )
            ))
