TODO:
- Comments on config.yaml
- Comments on polygons.yaml

# Non-rigid Multi-object Tracking
Azzolin Steve, Destro Matteo \
Signal, Image and Video \
Year 2020/21




## Get started

### Import the conda env
- Run `conda env create -f environment.yaml`.
- Run `conda activate name_of_the_environment`.
- **Note:** In case you want to run the Lin-pun tracker, you also need to compile some additional C libraries by running `cd prim && make`.

### Run sample
Simply run `main.py` to run the algorithm with the default parameters and configurations.

### Configurations
`config.yaml` contains all the configurable parameters for the maskers. The most important ones are:
- `input_video`: path of the video to use, relative to the root directory of the project.
- `masker`: identifier of the masker to be used, see section ["Maskers"](#maskers) for a list of the available maskers.
- `manual_roi_selection`: set to False to use the polygon specified in the `pts` parameter as initial selection. If set to True, the algorithm asks for a manual selection. 
- `show_masks`: Set to True to show the resulting mask while running the algorithm.

See the comments in `config.yaml` for more details and a description of the other parameters.




## Project structure

### Maskers
The folder `masker` contains separate classes for each algorithm proposal. The available maskers are:
- **BgSub** (`bg_subtractor_masker.py`): background subtractor.
- **LinPuntracker** (`lin_pun_tracker.py`): Lin-pun highly non-rigid object tracker.
- **OpticalFlow** (`optical_flow_masker.py`): Optical Flow and Convex Hull masker (OPCH).
- **SemiSupervised** (`semi_supervised_masker.py`): Pixel Classification (PC).
- **GrabCut** (`grab_cut.py`): GrabCut-based algorithm.

### Input/Output

### Benchmark
`benchmark.py` is a utility script to easily test different parameters combination automatically. It is useful in particular with the *SemiSupervised* tracker which has many hyper-parameters.

The benchmark uses `config_benchmark.yaml` as config file (see section [Configurations]("#configurations") for more details). For hyper-parameter testing, inside of `benchmark.py` two variables can be found:
- `VIDEOS`: the file name of the video files we want to run the benchmark on. Note that if you want to add a new video to the benchmark, you need to define the points of the selection masks in `polygons.yaml`.
- `HYPERPARAMS`: contains, for each hyper-parameter, a list of values to be tested, for which the script automatically generates and tests any possible combination.


The results are then saved into `benchmark_results.csv`, with the obtained benchmark and the processing time required for each video.