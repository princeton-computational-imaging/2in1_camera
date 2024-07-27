#!/bin/bash

# Saving and logging
RESULT_DIR=
PARAM=config/param_depth.py

PRETRAINED_DOE=
PRETRAINED_G=
EPOCHS=

APPLICATION=Depth

RECON_LOSS_WEIGHT=1

python train.py --train_G --result_path $RESULT_DIR --param_file $PARAM \
--n_epochs $EPOCHS --pretrained_DOE $PRETRAINED_DOE --pretrained_G $PRETRAINED_G --application $APPLICATION \
--recon_loss_weight $RECON_LOSS_WEIGHT --log_freq 500 --save_freq 1000