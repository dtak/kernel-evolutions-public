import json
from pathlib import Path
import os
import sys
import csv
import pickle
import trajectory_experiments
import pure_selection_experiments
import multitask_selection_experiments

from src.tools.experiments import get_real_data, get_toy_data, get_timeseries_data, get_path_from_setting, get_user_data, dir_to_args, get_multi_data

def args_to_filename(exp_kwargs): 
    string = ""
    i = 0
    for key, val in exp_kwargs.items(): 
        if i == 0: 
            string += "{}-{}".format(key, val)
        else: 
            string += "-{}-{}".format(key, val)
        i += 1

    return string + ".pickle" 

def run_experiment(method_name, exp_dir, exp_name, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    2. Get your experiment's parameters from 'exp_kwargs'
    3. Run your experiment
    4. Store the results however you see fit in 'exp_dir'
    '''
    
    outdir = "{}/{}/".format(exp_dir, exp_name)

    print('Running experiment {} with method {} output to {}...'.format(exp_name, method_name, outdir))
    
    # TOY KERNEL SELECTION
    elif method_name == "traj_kernel_selection_toy_full":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_toy_full(X_list, y_list, outdir = outdir, **exp_kwargs)
    elif method_name == "traj_kernel_selection_toy":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_toy(X_list, y_list, outdir = outdir, **exp_kwargs)
    elif method_name == "mh_kernel_selection_toy":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.mh(X_list_user, y_list_user, data = "toy", user = user, outdir = outdir, **exp_kwargs)
    elif method_name == "ard_kernel_selection_toy":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.ard(X_list_user, y_list_user, data = "toy", user = user, outdir = outdir, **exp_kwargs)
    elif method_name == "cluster_kernel_selection_toy":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.cluster_selection(X_list, y_list, outdir = outdir, data = "toy", **exp_kwargs)
    elif method_name == "stratified_kernel_selection_toy":
        X_list, y_list, users, exp_kwargs, _ = get_toy_data(method_name, exp_kwargs)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.stratified_selection(X_list, y_list, outdir = outdir, data = "toy", **exp_kwargs)
    # WINE
    elif method_name == "traj_kernel_selection_wine_full":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real_full(X_list, y_list, outdir = outdir, dataset = "wine", **exp_kwargs)
    elif method_name == "traj_kernel_selection_wine":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real(X_list, y_list, outdir = outdir, dataset = "wine", **exp_kwargs)
    elif method_name == "stratified_kernel_selection_wine":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.stratified_selection(X_list, y_list, outdir = outdir, data = "wine", **exp_kwargs)
    elif method_name == "cluster_kernel_selection_wine":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.cluster_selection(X_list, y_list, outdir = outdir, data = "wine", **exp_kwargs)
    elif method_name == "mh_kernel_selection_wine":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = False, shuffle = True)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.mh(X_list_user, y_list_user, user = user,data = "wine", outdir = outdir, **exp_kwargs)
    elif method_name == "ard_kernel_selection_wine":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/wine/", scaleall = False, shuffle = True)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.ard(X_list_user, y_list_user, data = "wine", user = user, outdir = outdir, **exp_kwargs)
    # CHLORIDES
    elif method_name == "cluster_kernel_selection_chlorides":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/chlorides/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.cluster_selection(X_list, y_list, outdir = outdir, data = "chlorides", **exp_kwargs)
    elif method_name == "traj_kernel_selection_chlorides_full":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/chlorides/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real_full(X_list, y_list, outdir = outdir, dataset = "chlorides", **exp_kwargs)
    elif method_name == "traj_kernel_selection_chlorides":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/chlorides/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real(X_list, y_list, outdir = outdir, dataset = "chlorides", **exp_kwargs)
    elif method_name == "stratified_kernel_selection_chlorides":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/chlorides/", scaleall = True, shuffle = True)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.stratified_selection(X_list, y_list, outdir = outdir, data = "chlorides", **exp_kwargs)
    elif method_name == "ard_kernel_selection_chlorides":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/chlorides/", scaleall = False, shuffle = True)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.ard(X_list_user, y_list_user, data = "chlorides", user = user, outdir = outdir, **exp_kwargs)
    # AIR
    elif method_name == "cluster_kernel_selection_air":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/air/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.cluster_selection(X_list, y_list, outdir = outdir, data = "air", **exp_kwargs)
    elif method_name == "traj_kernel_selection_air":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/air/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real(X_list, y_list, outdir = outdir, dataset = "air", **exp_kwargs)
    elif method_name == "traj_kernel_selection_air_full":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/air/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real_full(X_list, y_list, outdir = outdir, dataset = "air", **exp_kwargs)
    elif method_name == "stratified_kernel_selection_air":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/air/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.stratified_selection(X_list, y_list, outdir = outdir, data = "air", **exp_kwargs)
    elif method_name == "ard_kernel_selection_air":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/air/", scaleall = False, shuffle = False)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.ard(X_list_user, y_list_user, data = "air", user = user, outdir = outdir, **exp_kwargs)
    # POLLUTANTS
    elif method_name == "cluster_kernel_selection_pollutants":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/pollutants/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.cluster_selection(X_list, y_list, outdir = outdir, data = "pollutants", **exp_kwargs)
    elif method_name == "traj_kernel_selection_pollutants":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/pollutants/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real(X_list, y_list, outdir = outdir, dataset = "pollutants", **exp_kwargs)
    elif method_name == "traj_kernel_selection_pollutants_full":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/pollutants/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        trajectory_experiments.kernel_selection_real_full(X_list, y_list, outdir = outdir, dataset = "pollutants", **exp_kwargs)
    elif method_name == "stratified_kernel_selection_pollutants":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/pollutants/", scaleall = True, shuffle = False)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        multitask_selection_experiments.stratified_selection(X_list, y_list, outdir = outdir, data = "pollutants", **exp_kwargs)
    elif method_name == "ard_kernel_selection_pollutants":
        X_list, y_list, users, exp_kwargs = get_real_data(method_name, exp_kwargs, data_dir = REPO_DIR + "data/uci/pollutants/", scaleall = False, shuffle = False)
        user = exp_kwargs['user']
        X_list_user, y_list_user = get_user_data(user, X_list, y_list)
        outdir, exp_kwargs = get_path_from_setting(method_name, exp_kwargs, users, outdir)
        pure_selection_experiments.ard(X_list_user, y_list_user, data = "pollutants", user = user, outdir = outdir, **exp_kwargs)
    print('Results are stored in:', outdir)
    print('with hyperparameters', exp_kwargs)
    print('\n')

def main():
    assert(len(sys.argv) > 2)
    method_name = sys.argv[1]
    exp_dir = sys.argv[2]
    exp_name = sys.argv[3]
    exp_kwargs = json.loads(sys.argv[4])
    
    run(method_name, exp_dir, exp_name, exp_kwargs)


if __name__ == '__main__':
    main()
