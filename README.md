# RLBench in RLDS Format

## Environment Setup

1. Create a conda environment using environment files, e.g.:

```bash
conda env create -f rlds_dataset_builder/environment_ubuntu.yml
```

2. Install [RLBench](https://github.com/stepjam/RLBench?tab=readme-ov-file#install)

## Data Generation

The `ur5_data_generator.py` script in `rlbench_dataset` directory generates the data.
It is a slightly changed version of `dataset_generator.py` from the original RLBench repository.
It allows to generate data with UR5 robot.
Example usage:

```bash
python ur5_data_generator.py \
  --tasks place_shape_in_shape_sorter \
  --image_size 256 256 \
  --renderer opengl3 \
  --processes 4 \
  --episodes_per_task 10 \
  --variations 5 \
  --arm_max_velocity 1.0 \
  --arm_max_acceleration 4.0 \
  --robot_setup ur5
```

This should create a dataset in the `data/place_shape_in_shape_sorter` directory.

## Convertion to RLDS Format

To convert to RLDS format:
   1. Set `RLBENCH_GENERATED_DATASET_PATH` (eg. `data/place_shape_in_shape_sorter`), `ACTION_MODE` (chosen from `rlbench_dataset/action_modes.py` file), `PROPRIO_MODE` (chosedn from `rlbench_dataset/proprio_modes.py`) constants at the top of the `rlbench_dataset/rlbench_dataset.py` file.
   2. Change `VERSION` and `RELEASE_NOTES` constants in `RLBenchDataset` class in `rlbench_dataset/rlbench_dataset.py` file.
   3. Run `tfds build` inside the `rlbench_dataset` directory.
The converted dataset will be stored in the `tensorflow_datasets` directory in your home directory.

## Data Visualization

To visualize the data run:

```bash
python visualize_dataset.py rl_bench_dataset
```

or specify a version

```bash
python visualize_dataset.py rl_bench_dataset
```


## Notes and troubleshooting:

1. It can take up to several dozen minutes to generate the data.
For testing purposes you can reduce `variations` and `episodes_per_task`
2. If you wish, you can use a local, editable version of RLBench.
Simply run `pip install -e RLBench` instead of `pip install git+https://github.com/stepjam/RLBench.git`.
3. If you're getting

```bash
ImportError: libcoppeliaSim.so.1: cannot open shared object file: No such file or directory`
```

make sure that shell variables for Coppelia are exported (i.e. the ones exported during installation of RLBench):

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

---
---
---

# Archive

The text below is from the original repository.


# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for X-embodiment experiment integration.
It provides an example for converting a dummy dataset to RLDS. To convert your own dataset, **fork** this repo and 
modify the example code for your dataset following the steps below.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting your Own Dataset to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `example_dataset` to the name of your dataset (e.g. robo_net_v2), 
also change the name of `example_dataset_dataset_builder.py` by replacing `example_dataset` with your dataset's name (e.g. robo_net_v2_dataset_builder.py)
and change the class name `ExampleDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. RoboNetV2).

2. **Modify Features**: Modify the data fields you plan to store in the dataset. You can find them in the `_info()` method
of the `ExampleDataset` class. Please add **all** data fields your raw data contains, i.e. please add additional features for 
additional cameras, audio, tactile features etc. If your type of feature is not demonstrated in the example (e.g. audio),
you can find a list of all supported feature types [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?hl=en#classes).
You can store step-wise info like camera images, actions etc in `'steps'` and episode-wise info like `collector_id` in `episode_metadata`.
Please don't remove any of the existing features in the example (except for `wrist_image` and `state`), since they are required for RLDS compliance.
Please add detailed documentation what each feature consists of (e.g. what are the dimensions of the action space etc.).
Note that we store `language_instruction` in every step even though it is episode-wide information for easier downstream usage (if your dataset
does not define language instructions, you can fill in a dummy string like `pick up something`).

3. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g. 
by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.

4. **Modify Dataset Conversion Code**: Next, modify the function `_generate_examples()`. Here, your own raw data should be 
loaded, filled into the episode steps and then yielded as a packaged example. Note that the value of the first return argument,
`episode_path` in the example, is only used as a sample ID in the dataset and can be set to any value that is connected to the 
particular stored episode, or any other random value. Just ensure to avoid using the same ID twice.

5. **Provide Dataset Description**: Next, add a bibtex citation for your dataset in `CITATIONS.bib` and add a short description
of your dataset in `README.md` inside the dataset folder. You can also provide a link to the dataset website and please add a
few example trajectory images from the dataset for visualization.

6. **Add Appropriate License**: Please add an appropriate license to the repository. 
Most common is the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license -- 
you can copy it from [here](https://github.com/teamdigitale/licenses/blob/master/CC-BY-4.0).

That's it! You're all set to run dataset conversion. Inside the dataset directory, run:
```
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.


### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function 
takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
description of the desired output spec.

2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`

If the test passes successfully, you are ready to upload your dataset!

## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command 
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:
```
gcloud auth login
``` 
This will open a browser window that allows you to log into your Google account (if you're on a headless server, 
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address. 
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask 
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following 
command:
```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://xembodiment_data
``` 
This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/xembodiment_data).

The last step is to commit all changes to this repo and send Karl the link to the repo.

**Thanks a lot for contributing your data! :)**
