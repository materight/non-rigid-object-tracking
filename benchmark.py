import os, subprocess, shutil
import numpy as np
import itertools
import yaml

def computeBenchmark(mask, truth):
    ''' 
    Compute the intersection / union ratio between the resulting mask and the ground truth
    '''
    bmask, btruth = mask.astype(np.bool), truth.astype(np.bool)
    benchmark = np.sum(bmask & btruth) / np.sum(bmask | btruth)
    return benchmark

'''
Run main.py multiple times with different parameters, in order to try different hyperparameter values
'''
TMP_PATH = 'tmp'

with open('config_benchmark.yaml') as f:
    CONFIGS = yaml.full_load(f)

HYPERPARAMS = {
    'n_estimators': [0],
    'max_depth': [0],
    'n_components': [0, 1],
    'novelty_detection': [0],
    'over_segmentation': ['quickshift', 'felzenszwalb']
}

# Generate params list with all possible combinations
keys, values = zip(*HYPERPARAMS.items())
params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Delete older params
if os.path.exists(TMP_PATH):
    shutil.rmtree(TMP_PATH, ignore_errors=True)  
os.makedirs(TMP_PATH)

# Launch different subprocesses for each param combination
for i, params in enumerate(params_list):
    config_path = f'{TMP_PATH}/config{i}.yaml'
    # Create config file and run process
    with open(config_path, 'w') as config_file:
        configs = {**CONFIGS, 'params': params}
        yaml.dump(configs, config_file)
        # subprocess.Popen(f'python main.py {config_path}')