#!/bin/bash
# Saving and logging
RESULT_DIR=
PARAM=config/param_hdr.py

PRETRAINED_DOE=None
PRETRAINED_G=None
EPOCHS=100

APPLICATION=HDR

PSF_DESIGN=HDR_E2E
PSF_LOSS_WEIGHT=0

FAB_REG_WEIGHT=0
DOE_NOISE=0

RECON_LOSS_WEIGHT=1

python train.py --train_optics --train_G --result_path $RESULT_DIR --param_file $PARAM \
--n_epochs $EPOCHS --pretrained_DOE $PRETRAINED_DOE --pretrained_G $PRETRAINED_G --application $APPLICATION \
--PSF_design $PSF_DESIGN --fab_reg_weight $FAB_REG_WEIGHT --PSF_loss_weight $PSF_LOSS_WEIGHT \
--recon_loss_weight $RECON_LOSS_WEIGHT --DOE_phase_noise_scale $DOE_NOISE \
--log_freq 500 --save_freq 1000 