#!/bin/bash

# Saving and logging
RESULT_DIR=
PARAM=config/param_hs.py

PRETRAINED_DOE=
PRETRAINED_G=
EPOCHS=

APPLICATION=HS

RECON_LOSS_WEIGHT=1

python train.py --train_G --result_path $RESULT_DIR --param_file $PARAM \
--n_epochs $EPOCHS --pretrained_G $PRETRAINED_G --application $APPLICATION \
--pretrained_G $PRETRAINED_G --recon_loss_weight $RECON_LOSS_WEIGHT 