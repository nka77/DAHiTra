import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
#from apex import amp
import torch.cuda.amp as amp
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from adamw import AdamW
from torch.autograd import Variable

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timeit

from zoo.models import BASE_Transformer, Res34_Unet_Loc
from zoo.model_transformer_encoding import BASE_Transformer_UNet, Discriminator
from losses import dice_round, ComboLoss
from utils import *

import gc
torch.cuda.empty_cache()

model = ""
device = ('cuda' if torch.cuda.is_available() else 'cpu')

if model == "TUNet":
    print("UNet Transformer")
    model = BASE_Transformer_UNet(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).to(device)
    snapshot_name = 'BASE_UNet_Transformer_img1024_loc'
    print("snapshot_name ", snapshot_name, "with seg and cls headers and ce loss only on building")
    print("upsampling 1:3 with 50%")
    print("FIXED LOSS")
    snap_to_load = ''
else:
    print("Siamese ResNEt")
    model = Res34_Unet_Loc().to(device)
    snapshot_name = 'Res34_Unet_Double_img1024'
    snap_to_load = 'res34_loc_0_1_best'

## Variables
input_shape = (1024,1024)
crop_size = 1024
_thr = 0.3
cudnn.benchmark = True
batch_size = 1
val_batch_size = 1
train_dirs = ['../data/xbd/train']#, 'data/AOI3']
models_folder = 'weights'

all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if ('_pre_disaster.png' in f):# and (('hurricane-harvey' in f)):# | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f)):
            path_ = path.join(d, 'images', f)
            path_ = path_.replace('/', '//')
            all_files.append(path_)


## Creating data loader
class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = np.array(Image.open(fn))
        if random.random() > 0.8:
            img = np.array(Image.open(fn.replace('_pre_disaster', '_post_disaster')))

        msk0 = np.array(Image.open(fn.replace('/images/', '/masks/')))
        
        if random.random() > 0.5:
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]
        
        if random.random() > 0.7:
            imgs = [img]
            labels = [msk0]
            imgs = [TF.to_pil_image(img) for img in imgs]
            labels = [TF.to_pil_image(img) for img in labels]

            if random.random() > 0.3:
                imgs = [TF.hflip(img) for img in imgs]
                labels = [TF.hflip(img) for img in labels]

            if random.random() > 0.3:
                imgs = [TF.vflip(img) for img in imgs]
                labels = [TF.vflip(img) for img in labels]
            
            if random.random() > 0.3:
                x = random.randint(0, 200)
                y = random.randint(0, 200)
                imgs = [TF.resized_crop(img, x, y, crop_size-x, crop_size-y, (crop_size,crop_size)) for img in imgs]
                labels = [TF.resized_crop(img, x, y, crop_size-x, crop_size-y, (crop_size,crop_size)) for img in labels]

            if random.random() > 0.7:
                imgs = [transforms.ColorJitter(brightness=[0.8,1.2], contrast=[0.8,1.2], saturation=[0.8,1.2])(img) for img in imgs]
            
            msk0 = np.array(labels[0])
            img = np.array(imgs[0])


        msk = msk0[..., np.newaxis]
        msk = (msk > 127) * 1
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = np.array(Image.open(fn))
        msk0 = np.array(Image.open(fn.replace('/images/', '/masks/')))
        
        msk = msk0[..., np.newaxis]
        msk = (msk > 127) * 1
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


def validate(net, data_loader):
    dices0 = []
    _thr = 0.5

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].to(device)
            out = model(imgs)

            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

    d0 = np.mean(dices0)
    print("Val Dice: {}".format(d0))
    return d0


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].to(device)
        msks = sample["msk"].to(device)
        out = model(imgs)

        loss = seg_loss(out, msks)
        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))
        dices.update(dice_sc, imgs.size(0))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Dice {dice.avg:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    seed = 0 

    file_classes = []
    AOI_files = []
    for fn in tqdm(all_files):
        fl = np.zeros((4,), dtype=bool)
        msk1 = np.array(Image.open(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')))
        for c in range(1, 5):
            fl[c-1] = c in msk1
        file_classes.append(fl)
        if 'AOI' in fn:
            file_classes.append(fl)
    file_classes = np.asarray(file_classes)

    train_idxs0, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)

    np.random.seed(seed + 321)
    random.seed(seed + 321)

    train_idxs = []
    non_zero_bldg = 0
    non_zero_dmg = 0
    for i in train_idxs0:
        if file_classes[i, :].max():
            train_idxs.append(i)
            non_zero_bldg += 1
        if (random.random() > 0.5) and file_classes[i, 1:].max():
            train_idxs.append(i)
            non_zero_dmg += 1

    train_idxs = np.asarray(train_idxs)
    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size
    print(non_zero_bldg, non_zero_dmg, len(train_idxs), len(val_idxs))
    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=8, shuffle=False, pin_memory=False)

    optimizer = AdamW(model.parameters(), lr=0.00015, weight_decay=1e-6)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.6)
    
    if os.path.exists(path.join(models_folder, snap_to_load)):
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            k_ = 'module.'+ k
            if k_ in loaded_dict and sd[k].size() == loaded_dict[k_].size():
                sd[k] = loaded_dict[k_]
            else:
                print(k_, k, "failure")
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        del loaded_dict
        del sd
        del checkpoint

    if device == 'cuda':
        torch.cuda.empty_cache()
        model = nn.DataParallel(model).to(device)

    gc.collect()
    best_score = 0

    scaler = amp.GradScaler()
    seg_loss = ComboLoss({'dice': 1.0, 'focal': 10.0}, per_image=False).to(device)
    
    for epoch in range(100):
        train_epoch(epoch, model, optimizer, scheduler, train_data_loader)
        if epoch % 2 == 0:
            best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)
            torch.cuda.empty_cache()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

