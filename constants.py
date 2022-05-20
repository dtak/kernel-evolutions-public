import numpy as np

HYPER_PRIORS_FS = {
        'noise':{'mean': 0, 'var':2},
        'location':{'mean': 0, 'var': 0.1},
        'period': {'mean': np.log(20./4), 'var': 0.25},
        'lengthscale': {'mean': 0, 'var': 2}, 
        'amplitude': {'mean': 0, 'var': 2}
}

HYPER_PRIORS_TOY = {
        'noise':{'mean': 0, 'var': 2},
        'location':{'mean': 0, 'var': 0.1},
        'period': {'mean': np.log(20./4), 'var': 0.25},
        'lengthscale': {'mean': 0, 'var': 2}, 
        'amplitude': {'mean': 0, 'var': 2}
        }

TOY_INCLUSION_PROB = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
HYPER_PRIORS_SCALED = {
        'noise':{'mean': 0, 'var': 2},
        'location':{'mean': 0, 'var': 0.1},
        'period': {'mean': np.log(0.2), 'var': 0.25},
        'lengthscale': {'mean': np.log(0.2), 'var': 0.5},
        'amplitude': {'mean': 0, 'var': 2}
}

import src.tools.base_kernels as base

KERNEL_POOL = {
        'wine':base.create_wine_kernel_pool,
        'chlorides': base.create_chloride_kernel_pool, 
        'air': base.create_air_kernel_pool,
        'pollutants': base.create_air_kernel_pool,
}
