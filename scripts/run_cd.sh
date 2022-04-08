#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoints 
# data_name=xBDataset  # dataset name 
# dataset=xBDatasetMulti
# loss=focal
# n_class=5
# lr=0.0002
# lr_policy=multistep

data_name=LEVIR  # dataset name 
dataset=CDDataset
loss=ce
n_class=2
lr=0.005
lr_policy=linear

img_size=256
batch_size=8

max_epochs=200  #training epochs
net_G=newUNetTrans
#unet_coupled_trans_256
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_o5


split=train  # training txt
split_val=val  # validation txt
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_Crop

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}

# VAL Fixed and channel attention using diff 
# bit
# 0.5928 (at epoch 97)
# 0.5026 (at epoch 84)

# run 2: transformer 2, 5 and CA 2, 5 
# 0.5830 (at epoch 84)
# 0.5864 (epoch 94)

# previous archi
# 0.5767 (at epoch 95)

# siamese unet: 
# 0.5647 (at epoch 91)

# run 3: transformer 5 and CA 5 
# 0.5771 (at epoch 87)

# run 4: transformer 3, 5 and CA on 3, 5
# 0.5482 (at epoch 90)

# run 5: transformer 5 and CA on 3, 5
# 0.500 (at epoch 70)

# run 6: transformer 5 and CA on 2, 3, 4, 5
# 0.302

# run 7: transformer 3 and CA on 2, 3, 4, 5
# 0.393

# run 8: transformer 3 (followed by conv) and CA on 2, 5
# NW

# run 9: transformer 5 (followed by conv) and CA on 2, 3
# NW

# run 10: transformer 3 and 5 (followed by conv) 
# < 50

# run 11: transformer 3 and 5 (followed by conv) and CA 2
# < 50
