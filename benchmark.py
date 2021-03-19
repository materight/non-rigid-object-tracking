import os, subprocess, multiprocessing.pool, shutil
import numpy as np
import pandas as pd
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

def run_main(i, v, config_path, benchmark_out_path, results):
    process = subprocess.Popen(['python', 'main.py', config_path, benchmark_out_path], stdout=FNULL)
    process.wait()
    pbar.update(n=1)
    with open(benchmark_out_path) as benchmark_file:
        benchmark, time = map(float, benchmark_file.readline().split(';'))
        # results[i][v] = f'{benchmark:.2f} ({time:.2f}s)'
        results[i][v+'_benchmark'] = benchmark
        results[i][v+'_time'] = time

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

    VIDEOS = ['soldier', 'frog', 'worm', 'bmx']
    HYPERPARAMS = {
        'n_estimators': [20, 30],
        'max_depth': [7,10],
        'n_components': [1],
        'novelty_detection': [True, False],
        'over_segmentation': ['quickshift', 'felzenszwalb'],
        'features': ['8 hsv_lab', '6 lab'],
        "dilation_kernel": [7],
        "prior_weight": [0.1, 0.0]
    }

    # Generate params list with all possible combinations
    keys, values = zip(*HYPERPARAMS.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Delete older params
    if os.path.exists(TMP_PATH):
        shutil.rmtree(TMP_PATH, ignore_errors=True)  
    os.makedirs(TMP_PATH)

    # Launch different subprocesses for each video and param combination
    NCPU = None # Number of parallel processes to spawn. None to use the available number of core
    tp = multiprocessing.pool.ThreadPool(processes=NCPU)
    pbar = tqdm(total=len(params_list) * len(VIDEOS))
    results = {}
    for i, params in enumerate(params_list):
        results[i] = params.copy()
        for v in VIDEOS:
            configs = {
                **CONFIGS, 
                'input_video': f'{VIDEOS_PATH}/{v}.mp4',
                'input_truth': f'{TRUTH_PATH}/{v}.mp4',
                'params': params,
                **POLYGONS[v]
            }
            config_path = f'{TMP_PATH}/config-{i}-{v}.yaml'
            benchmark_out_path = f'{TMP_PATH}/results-{i}-{v}.csv'
            with open(config_path, 'w') as config_file:
                yaml.dump(configs, config_file, sort_keys=False)
            tp.apply_async(run_main, (i, v, config_path, benchmark_out_path, results))
    tp.close() # No more thread to run
    tp.join() # Wait for all the thread to finish
    pbar.close()
    # Save results
    results = pd.DataFrame.from_dict(results, orient='index')
    results['avg_benchmark'] = results[[f'{v}_benchmark' for v in VIDEOS]].mean(axis=1) # Compute average benchmark column
    #with open('benchmark_results.csv', 'w') as f: results.to_markdown(f, index=False, tablefmt='github')
    results.to_csv('benchmark_results.csv', index=False)