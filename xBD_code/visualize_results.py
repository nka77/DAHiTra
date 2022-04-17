import os
from torch import nn
import torch
from torch.autograd import Variable
from zoo.models import BASE_Transformer, Res34_Unet_Loc, Res34_Unet_Double
from zoo.model_transformer_encoding import BASE_Transformer_UNet
from torchvision import utils

from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
import numpy as np
np.random.seed(1)
import random
random.seed(1)

from tqdm import tqdm
import timeit
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from utils import *

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = '/scratch/nka77/DATA/test/images_val/'
mask_dir = '/scratch/nka77/DATA/test/masks_val/'
models_folder = '/scratch/nka77/xview_first/weights'

_thr = [0.38, 0.13, 0.14]
t0 = timeit.default_timer()

model_str = "TUNet"

if model_str == "TUNet":
    model = BASE_Transformer_UNet(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                              with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).cuda()
    snap_to_load = 'BASE_UNet_Transformer_V4_lr001_onlyB_3_img512_lossv1'
    sub_folder = '/scratch/nka77/xview_first/Mar27/TUNet/'
elif model_str == "BiT":
   model = BASE_Transformer(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                              with_pos='learned', enc_depth=1, dec_depth=8).cuda()
   snap_to_load = 'BiT_F1update'
   sub_folder = '/scratch/nka77/xview_first/Mar27/BiT/'
else:
   model = Res34_Unet_Double().cuda()
   snap_to_load = 'res34_cls2_1_tuned_best'
   sub_folder = '/scratch/nka77/xview_first/Mar27/ResNet/'

makedirs(sub_folder, exist_ok=True)

all_files = []
for f in tqdm(sorted(listdir(test_dir))):
    all_files.append(f)


loc_snap_to_load = 'res34_loc_0_1_best'
loc_model = Res34_Unet_Loc(pretrained=False).cuda()
loc_model = nn.DataParallel(loc_model).cuda()
print("=> loading checkpoint '{}'".format(loc_snap_to_load))
checkpoint = torch.load(path.join(models_folder, loc_snap_to_load), map_location='cpu')
loaded_dict = checkpoint['state_dict']
sd = loc_model.state_dict()
for k in loc_model.state_dict():
    if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        sd[k] = loaded_dict[k]
loaded_dict = sd
loc_model.load_state_dict(loaded_dict)
print("loaded checkpoint '{}' (epoch {}, best_score {})"
        .format(loc_snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
loc_model.eval()


model = nn.DataParallel(model).cuda()
print("=> loading checkpoint '{}'".format(snap_to_load))
checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
loaded_dict = checkpoint['state_dict']
sd = model.state_dict()
for k in model.state_dict():
    if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        sd[k] = loaded_dict[k]
loaded_dict = sd
model.load_state_dict(loaded_dict)
print("loaded checkpoint '{}' (epoch {}, best_score {})"
        .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
model.eval()

def get_loc_mask(img):
    model = loc_model
    img = preprocess_inputs(img)
    inp = []
    inp.append(img)
    inp.append(img[::-1, ...])
    inp.append(img[:, ::-1, ...])
    inp.append(img[::-1, ::-1, ...])
    inp = np.asarray(inp, dtype='float')
    inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
    inp = Variable(inp).cuda()

    pred = []
    msk = model(inp)
    msk = torch.sigmoid(msk)
    msk = msk.cpu().numpy()
    pred.append(msk[0, ...])
    pred.append(msk[1, :, ::-1, :])
    pred.append(msk[2, :, :, ::-1])
    pred.append(msk[3, :, ::-1, ::-1])

    pred_full = np.asarray(pred).mean(axis=0)
    msk = pred_full * 255
    msk = msk.astype('uint8').transpose(1, 2, 0)[..., 0]
    return msk

def get_dmg_mask(img1, img2):
    img = np.concatenate([img1, img2], axis=2)
    img = preprocess_inputs(img)

    inp = []
    inp.append(img)
    inp.append(img[::-1, ...])
    inp.append(img[:, ::-1, ...])
    inp.append(img[::-1, ::-1, ...])
    inp = np.asarray(inp, dtype='float')
    inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
    inp = Variable(inp).cuda()

    pred = []
    msk = model(inp)
    msk = torch.sigmoid(msk)
    msk = msk.cpu().numpy()
    
    pred.append(msk[0, ...])
    pred.append(msk[1, :, ::-1, :])
    pred.append(msk[2, :, :, ::-1])
    pred.append(msk[3, :, ::-1, ::-1])

    pred_full = np.asarray(pred).mean(axis=0)
    
    # msk = pred_full * 255
    msk = pred_full.astype('uint8').transpose(1, 2, 0)
    return msk


# def make_numpy_grid(tensor_data, pad_value=0,padding=0):
#     # tensor_data = tensor_data.detach()
#     vis = utils.make_grid(torch.tensor(tensor_data)*255, pad_value=pad_value,padding=padding)
#     if vis.shape[2] == 1:
#         vis = np.stack([vis, vis, vis], axis=-1)
#     return vis


# def de_norm(tensor_data):
#     return tensor_data * 0.5 + 0.5

def assign_color(img):
    color_dict = {0:[0,0,0],
                    1:[0,255,0],
                    2:[0,255,255],
                    3:[0,127,255],
                    4:[0,0,255]}
    m,n = img.shape
    im_color = np.array([color_dict[img[i,j]] for i in range(m) for j in range(n)]).astype(np.uint8)
    img = im_color.reshape([m,n,3])
    return img


with torch.no_grad():
    for f in tqdm(sorted(listdir(test_dir)[100:150])):
        if '_pre_' in f:
            fn = path.join(test_dir, f)
            #print(fn)
            img1 = cv2.imread(fn, cv2.IMREAD_COLOR)[:512,:512,:]
            img2 = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)[:512,:512,:]
            grnd_mask = cv2.imread(fn.replace('_pre_', '_post_').replace('images_val', 'masks_val'), cv2.IMREAD_UNCHANGED)[:512,:512]
            
            img = np.concatenate([img1, img2], axis=2)
            img = preprocess_inputs(img)

            inp = []
            inp.append(img)
            inp.append(img[::-1, ...])
            inp.append(img[:, ::-1, ...])
            inp.append(img[::-1, ::-1, ...])
            inp = np.asarray(inp, dtype='float')
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp).cuda()

            pred = []
            msk = model(inp)
            msk = torch.sigmoid(msk)
            msk = msk.cpu().numpy()
            
            pred.append(msk[0, ...])
            pred.append(msk[1, :, ::-1, :])
            pred.append(msk[2, :, :, ::-1])
            pred.append(msk[3, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0)
            
            msk = pred_full * 255
            msk = msk.astype('uint8').transpose(1, 2, 0)
            msk_dmg = msk[..., 1:].argmax(axis=2) + 1

            loc_preds = get_loc_mask(img1)/255
            msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')

            # msk_dmg = msk_dmg * msk_loc

            out_dmg = assign_color(msk_dmg)
            grnd_mask = assign_color(grnd_mask)

            visual_grid = np.zeros([512,512*4,3])
            visual_grid[:,:512,:] = img1
            visual_grid[:,512:1024,:] = img2
            visual_grid[:,1024:1536,:] = grnd_mask
            visual_grid[:,1536:,:] = out_dmg

            filename = "outputs/" + model_str + "_" + f.replace('_pre_', '_vis').replace('_part1.png', '.png')
            cv2.imwrite(filename, visual_grid, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        

elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))

