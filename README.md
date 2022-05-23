# The Trajectory Evolution Model

## Explanation of Directories/files
* **data/**: stores the data used in the experiments.
    *  **toy/**: when toy data is generated by the `get_toy_data` function in `src/tools/experiments.py`, it also gets dumped here as a `.pickle` file
    * **uci/**: UCI data sets used in the experiments.  
* **experiment_settings/**: the parameters that are set for each data set, and each method. 
* **results/**: results of experiments are automatically created and stored here. 
* **src/**: source code. Top level directory contains all of the model classes (for Evolution/Final, Stratified, Memoryless, and ARD)
    * **tools/**: various helper functions for the experiments
        * `base_kernels.py`: defines the candidate kernel pool for each data set. The candidate kernels are usually each kernel function applied to each dimension of the data. 
        * `experiments.py`: helper functions for setting up the experiments

## Running an experiment 
### Meta-training
Run an experiment using the following command: `python run.py -m <method> -d <data>`.
For example, the `python run.py -m evolution -d toy` will run meta-training for the Evolution method on the synthetic data set. 
* The training results are stored in the following: `results/<data>/<method>/<experiment_directory>/`
    * For Evolution, Stratified, and Final, the results are `.pickle` files of the meta-model at each iteration. 
    * For Memoryless and ARD, results are further separated by user. 
* The experiment parameters can be found in `experiment_settings/<data>`. Adjusting these parameters will also change the corresponding experiment directory in which the results are output to. 
### Meta-testing
For a model at any given point in training, you can call the function in `meta_testing.py` to select models on the X, y data of a test user.   