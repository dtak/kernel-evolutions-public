from src.tools.experiments import get_data_stats
def select_models(X_list, y_list, model, method = "evolution", learn_hypers = False, validation = None): 
    # Return dict of models where models[(m, t)] = model
    # Return times too
    M, T = get_data_stats(X_list)
    kernels = {}
    times = {}
    if method == "evolution": 
        model._form_leaves()
        X_train_lists = [[X_list[(m, t)] for t in range(T[m])] for m in range(M)]
        y_train_lists = [[y_list[(m, t)] for t in range(T[m])] for m in range(M)]
        for m in range(M): 
            X_train_list = [X_list[(m, t)] for t in range(T[m])]
            y_train_list = [y_list[(m, t)] for t in range(T[m])]
            start = time.time()
            selected_kernels = model.predict_trajectory(X_train_list, y_train_list, threshold = 1, prioritize_leaves = False, learn_hypers = learn_hypers, validation = validation)  
            end = time.time()
            
            for t in range(T[m]): 
                kernels[(m, t)] = selected_kernels[t]
                times[(m, t)] = (end - start)/T[m]
    elif method == "final": 
        model._form_leaves()
        for m in range(M): 
            for t in range(T[m]): 
                start = time.time()
                kernel, _ = model.select_leaf(X_list[(m, t)], y_list[(m, t)], learn_hypers = learn_hypers, validation = validation)
                end = time.time()
                times[(m, t)] = end - start
                kernels[(m, t)] = kernel
    elif method == "stratified": 
        for m in range(M): 
            print("user ", m)
            for t in range(T[m]): 
                start = time.time()
                kernel = model.select_model(X_list[(m, t)], y_list[(m, t)], t, learn_hypers = learn_hypers, validation = validation) 
                end = time.time()
                times[(m, t)] = end - start
                kernels[(m, t)] = kernel
    elif method == "memoryless":
        for m in range(M):
            for t in range(T[m]):
                kernel = model[m][t].model
                kernels[(m, t)] = kernel
                times[(m, t)] = model[m][t].runtime

    elif method == "ard":
        for m in range(M):
            for t in range(T[m]): 
                kernel = {"structure": model[m][t], "noise": model.original_noise}
                #kernel['structure'].active_dims = [0]
                kernels[(m, t)] = kernel     
                times[(m, t)] = model[m][t]['runtime']
    return kernels, times       
