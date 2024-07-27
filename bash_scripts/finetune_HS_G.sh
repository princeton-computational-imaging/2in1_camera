#!/bin/bash

# Saving and logging
RESULT_DIR=
PARAM=config/param_hs.py

PSF_FILE=
PRETRAINED_G=
EPOCHS=100

APPLICATION=HS

RECON_LOSS_WEIGHT=1

python train.py --train_G --result_path $RESULT_DIR --param_file $PARAM \
--n_epochs $EPOCHS --pretrained_G $PRETRAINED_G --application $APPLICATION \
--PSF_file $PSF_FILE --recon_loss_weight $RECON_LOSS_WEIGHT 