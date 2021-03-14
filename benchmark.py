import os, subprocess, multiprocessing.pool, shutil
import numpy as np
import itertools
from tqdm import tqdm
import yaml

def computeBenchmark(mask, truth):
    ''' 
    Compute the intersection / union ratio between the resulting mask and the ground truth
    '''
    bmask, btruth = mask.astype(np.bool), truth.astype(np.bool)
    benchmark = np.sum(bmask & btruth) / np.sum(bmask | btruth)
    return benchmark

if __name__ == "__main__":
    '''
    Run main.py multiple times with different parameters, in order to try different hyperparameter values
    '''
    FNULL = open(os.devnull, 'w')
    TMP_PATH = 'tmp'
    VIDEOS_PATH = 'output/Video/SegTrack2/Video'
    TRUTH_PATH = 'output/Video/SegTrack2/Truth'

    with open('config_benchmark.yaml') as f:
        CONFIGS = yaml.full_load(f)

    with open('polygons.yaml') as f:
        POLYGONS = yaml.full_load(f)

    VIDEOS = ['soldier']
    HYPERPARAMS = {
        'n_estimators': [10],
        'max_depth': [5],
        'n_components': [1],
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

    # Launch different subprocesses for each video and param combination
    tp = multiprocessing.pool.ThreadPool()
    pbar = tqdm(total=len(HYPERPARAMS) * len(VIDEOS))
    for i, params in enumerate(params_list):
        for v in VIDEOS:
            configs = {
                **CONFIGS, 
                'input_video': f'{VIDEOS_PATH}/{v}.mp4',
                'input_truth': f'{TRUTH_PATH}/{v}.mp4',
                'params': params,
                **POLYGONS[v]
            }
            config_path = f'{TMP_PATH}/config{i}.yaml'
            benchmark_out_path = f'{TMP_PATH}/results{i}.csv'
            with open(config_path, 'w') as config_file:
                yaml.dump(configs, config_file, sort_keys=False)
                def run():
                    process = subprocess.Popen(['python', 'main.py', config_path, benchmark_out_path], stdout=FNULL)
                    process.wait()
                    pbar.update(n=1)
                tp.apply_async(run)
    pbar.close()