TODO:
- Comments on config.yaml


# Non-rigid Multi-object Tracking
Project on non-rigid multi-object tracking for the the Signal, Image and Video course (2020/21)

## Get started



### Import the conda env
- Open the Anaconda shell
- Run *conda env create -f environment.yml*
- Run *conda activate name_of_the_environment*
- Open your favorite editor using the just-created environment (for example, fire *code .* in the same Anaconda shell)

### Run sample
Simply run `main.py` to ru the algorithm with the default parameters and configurations.

### Configurations
`config.yaml` contains all the configurable parameters for the maskers. The most important ones are:
- `input_video`: 
- `masker`: See section ["Maskers"](#maskers) for a list of the available maskers.
- `manual_roi_selection`:
- `show_masks`: 

See the comments in `config.yaml` for more details and a description of the other parameters.


### Note: compile C libraries
In the case you want to run the `LinPuntracker`, you need to compile some additional C libraries. To do so, run:
- `cd prim`
- `make`


## Project structure


### Maskers
The folder `masker` contains separate classes for each algorithm proposal. The available maskers are:
- **BgSub** (`bg_subtractor_masker.py`): background subtractor
- **LinPuntracker** (`lin_pun_tracker.py`): Lin-pun highly non-rigid object tracker
- **OpticalFlow** (`optical_flow_masker.py`): Optical Flow and Convex Hull masker (OPCH)
- **SemiSupervised** (`semi_supervised_masker.py`): Pixel Classification (PC)
- **GrabCut** (`grab_cut.py`): GrabCut-based algorithm

### Benchmark
