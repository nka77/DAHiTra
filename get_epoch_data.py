

# file = "checkpoints/CD_base_transformer_pos_s4_dd8_LEVIR_b8_lr0.01_train_val_200_linear/log.txt"

# with open(file) as f:
#     with open("BiT_accuracy")
#     line = f.readline()
#     while (line):
#         if line.startswith("Is_training: True. Epoch"):
#             print(line.split("= ")[1])
#         line = f.readline()

from models.networks import UNet_Change_Transformer, BASE_Transformer

model = UNet_Change_Transformer()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)


model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)



