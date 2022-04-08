#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=unet_coupled_two_trans_256
split=test
project_name=CD_unet_coupled_two_trans_256_LEVIR_b8_lr0.001_train_val_200_linear_focal_100wt
checkpoint_name=best_ckpt.pt

# CD_unet_coupled_two_trans_256_LEVIR_b8_lr0.001_train_val_200_linear_focal_100wt
# CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_train_val_200_linear_focal
# CD_unet_coupled_trans_256_LEVIR_b8_lr0.01_train_val_200_linear_focal/CD_unet_coupled_trans_256_LEVIR_b8_lr0.01_train_val_200_linear_focal

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


