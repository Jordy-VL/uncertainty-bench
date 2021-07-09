```python
__author__ = "Jordy Van Landeghem"
__copyright__ = "Copyright (C) 2020 Jordy Van Landeghem"
__license__ = "AGPL v3"
__version__ = "3.0"
```

# Benchmarking Scalable Predictive Uncertainty in Text Classification

📋 This repository is the official implementation of [Benchmarking Scalable Predictive Uncertainty in Text Classification](). 
The original library is called `arkham`, for continued development I kept this framework structure for the opensource version as well.


## Requirements & Setup

### Requirements

To install requirements via `apt` and `poetry/pip`:

```sh
git clone git@github.com:Jordy-VL/uncertainty-bench.git  # -tmp if reviewing PR
# optional, yet preferred, else cd uncertainty-bench/src
cd uncertainty-bench/src

sudo apt-get install python3-pip
sudo apt-get install python3-venv
sudo pip3 install virtualenv
sudo pip3 install virtualenvwrapper
mkvirtualenv -p /usr/bin/python3.6 -a . ENVname

#Using poetry and the readily defined pyproject.toml, we will install all required packages
pip3 install poetry
workon ENVname; poetry install
```

```python
## optionally test tensorflow (gpu) installation in interactive session
python3
import tensorflow as tf
tf.constant("hello TensorFlow!")
```

Main requirements are: 
* TensorFlow 2
* Sacred (for experiment configuration)
* wandb (for experiment tracking) [default: disabled if you do not provide an experiment name with `-n expname`]

### Setup

> :nut_and_bolt: `src/arkham/default_config.py` contains all default paths for saving/loading data/models

* `SAVEROOT` as destination for saving model artefacts (default: `uncertainty-bench/models`)
* `DATAROOT` as source for "dataset" directories (default: `uncertainty-bench/datasets`)
* `MODELROOT` as destination for re-loading models for evaluation

> :warning: You can skip this step if you do not want to set your custom absolute paths! The previous should run out-of-the-box

Add a `configfile.py` in `src/arkham` with the previous *GLOBALS*


#### Test your setup and all implemented uncertainty methods :sunglasses:

> :open_hands: we provide a dummy data set (sampled from imdb movie review data) to test your setup
> Be sure to put it under DATAROOT/mini_imdb if you changed the default_config

```sh
./test_setup.sh
```

> :boom: if this runs without issue, training - evaluating - saving and loading models; then you are all set up!

### Configs

In Sacred, configs are used to detail hyperparameter configs to the framework. Based on certain switches, different models and uncertainty methods will be activated.
Additionally, from the commandline, you can override a certain parameter. For example runs, see `benchmark_runs.md`.

#### To test the default setup:

```sh
python3 arkham/Bayes/Quantify/experiment.py with clf_default "identifier=mini_imdb" "model=cnn_baseline" "steps_per_epoch=None" epochs=1
```

## Repository structure

The most important implementations and helper functions are in these subdirectories:

*  `arkham/arkham/Bayes/Quantify`
*  `arkham/arkham/utils`
*  `arkham/arkham/Bayes/GP`
*  `arkham/arkham/Bayes/MCMC`

> :thought_balloon: All model training and evaluation starts from `arkham/arkham/Bayes/Quantify/experiment.py`


## Training

To train the model(s) in the paper, run this command:

```train
python3 experiment.py with clf_default identifier=<path_to_data> 
```

> 📋Describes how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate a trained model on its testset, run:

```eval
python3 evaluate.py <path_to_model>
```

This will write out an eval.pickle to <path_to_model> containing all the raw data to generate statistics and visualizations


To evaluate and compare multiple models for a dataset:

```compare
python3 compare.py <path_to_model> <identifier>
#e.g. Reuters
```

Output (shortened) looks like the following example: 
```
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
|                  version                  | accuracy |   mse    | F1 (micro) | F1 (macro) |  NLL   |  ECE   | Brier  | µ_conf | µ_entropy | µ_model | µ_data |
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
|      <path-to-model>_mc                   |  0.724   | 407.1914 |   0.7239   |   0.706    | 1.2844 | 0.0683 | 0.3982 | 0.6666 |   2.1333  |  0.0006 | 0.0094 |
|      <path-to-model>_nonbayesian          |  0.7233  | 423.0255 |   0.7229   |   0.7044   | 1.3522 | 0.1195 | 0.4106 | 0.609  |   2.7714  |   0.0   | 0.0181 |
+-------------------------------------------+----------+----------+------------+------------+--------+--------+--------+--------+-----------+---------+--------+
```

To run evaluations on a model with Out-of-Distribution data: 
```ood
python3 ood.py <path_to_model>
```


### Advanced Instructions

#### Adding a new dataset

Any new dataset should be a subdirectory of DATAROOT, with an identifier, which triggers the appropriate dataloader in `data.py`. 
There are plently examples there. The subdirectory itself should contain any one of three labelsets `train`, `dev` and/or `test`.
We include support for `flair`, `tensorflow-datasets` and `huggingface-nlp` datasets as examples.

#### Adding a new uncertainty method

Highly dependent on what architectural, optimization or additional hyperparameter changes are required.
Generally, add a standalone .py file with the `tf.keras.model` with `tf.keras.optimizers.optimizerV2`, callbacks, loss and compilation code. 
All other functionality should already run out-of-the-box.
Ideally, you include a new model switch in `experiment.py`, based on the `model_identifier` parameter in the default or a new config.
In order to load the model, this new object creation code should be imported in `model_utils.py`.


### More details [TBD]

Link to paper:


#### Reproducing main results
> 📋Describes how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


Run the commands in `benchmark_runs.md`


## Citing

If you use any part of this code in your research, please cite it using the following BibTex entry
```cite
@inproceedings{VanLandeghem2020,
  TITLE = {Benchmarking Scalable Predictive Uncertainty in Text Classification},
  AUTHOR = {Van Landeghem, Jordy and Blaschko, Matthew B. and Anckaert, Bertrand and Moens, Marie-Francine},
  BOOKTITLE = {Submitted to Journal of Machine Learning Research},
  YEAR = {2020}
}
}
```
