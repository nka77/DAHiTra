#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=newUNetTrans
split=test
project_name=CROP_newUNetTrans_LEVIR_b8_lr0.001_train_val_200_linear_ce_smoothen
checkpoint_name=best_ckpt.pt

#newUNetTrans
#CROP_newUNetTrans_LEVIR_b4_lr0.001_train_val_200_linear_ce

#changeFormerV6
#CROP_changeFormerV6_LEVIR_b4_lr0.0001_train_val_200_linear_ce

#base_transformer_pos_s4_dd8
#CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.0001_train_val_200_linear_ce_Resize

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


