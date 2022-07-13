#!/usr/bin/env bash

gpus=-1
checkpoint_root=checkpoints 
data_name=LEVIR
dataset=CDDataset
loss=ce
n_class=2
lr=0.001
lr_policy=linear

img_size=256
batch_size=2

max_epochs=200 
net_G=newUNetTrans

split=train  
split_val=val  
project_name=CROP_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_ce_smoothen

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}



# Different models trained:
#changeFormerV6
#unet_coupled_trans_256
#base_transformer_pos_s4_dd8

# For xBD dataset:
# data_name=xBDataset 
# dataset=xBDatasetMulti
# loss=focal
# n_class=5
# lr=0.0002
# lr_policy=multistep
