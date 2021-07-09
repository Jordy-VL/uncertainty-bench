#!/bin/bash
RUN=./arkham/Bayes/Quantify

dataset="mini_imdb" #dummy data
common="identifier=$dataset epochs=2 "steps_per_epoch=None" seed=42"
nodropout_baseline="dropout_nonlinear=0 dropout=0 dropout_concrete=False weight_decay=0 use_aleatorics=False"

# default
python3 $RUN/experiment.py with clf_default $common $nodropout_baseline

# MC Dropout
python3 $RUN/experiment.py with clf_default $common dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=1e-4 use_aleatorics=False

# Heteroscedastic model; requires to backtrack to TF2.2, disabled for now
#python3 $RUN/experiment.py with clf_default $common dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=False weight_decay=1e-4 use_aleatorics=True

# Concrete Dropout
python3 $RUN/experiment.py with clf_default $common dropout_nonlinear=0.5 dropout=0.5 dropout_concrete=True weight_decay=1e-4 use_aleatorics=False

# Deep Ensemble (M=5)
python3 $RUN/experiment.py with clf_default $common $nodropout_baseline ensemble=2

# SNGP (norm multiplier 1 for dense, 6 for convolution; with GP output layer)
python3 $RUN/experiment.py with clf_default SNGP_default $common $nodropout_baseline max_document_len=100 spec_norm_multipliers="[1,6]" use_gp_layer=True

# cSGCMMC (sample 1 model over 2 cycles, which each last 3 epochs)
python3 $RUN/experiment.py with clf_default cSGMCMC_clf $common $nodropout_baseline alpha=0.9 learning_rate=0.25 epochs=6 cycles=2 posterior_sampling=1