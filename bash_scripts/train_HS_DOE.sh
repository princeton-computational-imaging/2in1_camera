#!/bin/bash

# Saving and logging
RESULT_DIR=
PARAM=config/param_hs.py

PRETRAINED_DOE=
EPOCHS=
APPLICATION=HS

PSF_DESIGN=HS_rainbow
PSF_LOSS_WEIGHT=1
PSF_LOSS_RADIUS=2

FAB_REG_WEIGHT=0.1
DOE_NOISE=0.1

RECON_LOSS_WEIGHT=0

python train.py --train_optics --result_path $RESULT_DIR --param_file $PARAM \
--n_epochs $EPOCHS --pretrained_DOE $PRETRAINED_DOE --application $APPLICATION \
--fab_reg_weight $FAB_REG_WEIGHT --PSF_loss_weight $PSF_LOSS_WEIGHT --psf_loss_radius $PSF_LOSS_RADIUS \
--PSF_design $PSF_DESIGN --recon_loss_weight $RECON_LOSS_WEIGHT --DOE_phase_noise_scale $DOE_NOISE 