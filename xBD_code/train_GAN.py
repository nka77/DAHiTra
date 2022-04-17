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
from losses import dice_round, ComboLoss

from tqdm import tqdm
import timeit
#import cv2

from zoo.models import BASE_Transformer, Res34_Unet_Double
from zoo.model_transformer_encoding import BASE_Transformer_UNet, Discriminator


from torch.autograd import Variable
from PIL import Image

model = "TUNet"

if model == "TUNet":
    print("UNet Transformer")
    model = BASE_Transformer_UNet(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                              with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).cuda()
    snapshot_name = 'BASE_UNet_Transformer_img512_lossv7_AOI_GAN'
    print("snapshot_name ", snapshot_name, "with seg and cls headers and ce loss only on building")
    print("Loss only building patch lr:0.001 Seg weights: loss_seg = loss0")
    print("CE weights_ = torch.tensor([0.001,0.10,1.5,1.5,1.5])")
    print("upsampling 1:3 with 50%")
    print("GAN Loss")
    snap_to_load = 'res34_loc_0_1_best'

elif model == "BiT":
    print("BiT ....")
    model = BASE_Transformer(input_nc=3, output_nc=5, token_len=4, resnet_stages_num=4,
                              with_pos='learned', enc_depth=1, dec_depth=8).cuda()
    snapshot_name = 'BiT_lossv2'
    print("snapshot_name ", snapshot_name)
    print("Loss only building patch lr:0.001 Seg weights: loss_seg = loss0 ")
    print("CE weights_ = torch.tensor([0.001,0.10,1.5,1.0,1.5])")
    print("reduced upsampling of images 1 and 3")
    snap_to_load = 'res34_loc_0_1_best'

else:
    model = Res34_Unet_Double().cuda()
    snapshot_name = 'Res34_Unet_Double_img512_lossv5_AOI'
    snap_to_load = 'res34_loc_0_1_best'


#from imgaug import augmenters as nniaa
from utils import *
#from scikitimage.morphology import square, dilation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gc
torch.cuda.empty_cache()

#cv2.setNumThreads(0)
#cv2.ocl.setUseOpenCL(False)

train_dirs = ['../DATA/train', '../DATA/tier3', '../DATA/AOI3']
models_folder = 'weights'

input_shape = (1024,1024)
crop_size = 512
_thr = 0.3
cudnn.benchmark = True
batch_size = 8
val_batch_size = 8

all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if ('_pre_disaster.png' in f):# and (('hurricane-harvey' in f)):# | ('hurricane-michael' in f) | ('mexico-earthquake' in f) | ('tuscaloosa-tornado' in f)):
            all_files.append(path.join(d, 'images', f))


valid = Variable(torch.ones((batch_size, 1000)), requires_grad=False).cuda()
fake = Variable(torch.zeros((batch_size, 1000)), requires_grad=False).cuda()


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        #self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = np.array(Image.open(fn))
        img2 = np.array(Image.open(fn.replace('_pre_disaster', '_post_disaster')))

        msk0 = np.array(Image.open(fn.replace('/images/', '/masks/')))
        lbl_msk1 = np.array(Image.open(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')))
        
        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)

        img1 = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        lbl_msk1 = lbl_msk1[y0:y0+crop_size, x0:x0+crop_size]

        if random.random() > 0.7:
            imgs = [img1, img2]
            labels = [msk0, lbl_msk1]
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
            
            msk0, lbl_msk1 = np.array(labels[0]), np.array(labels[1])
            img1, img2 = np.array(imgs[0]), np.array(imgs[1])

        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk[..., 0] = False
        '''msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 3] = dilation(msk[..., 3], square(5))
        msk[..., 4] = dilation(msk[..., 4], square(5))'''
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = True
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = np.concatenate([img1, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.tensor(img.transpose((2, 0, 1))).float()
        msk = torch.tensor(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
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
        img2 = np.array(Image.open(fn.replace('_pre_disaster', '_post_disaster')))

        # msk_loc = cv2.imread(path.join(loc_folder, '{0}.png'.format(fn.split('/')[-1].replace('.png', '_part1.png'))), cv2.IMREAD_UNCHANGED) > (0.3*255)

        msk0 = np.array(Image.open(fn.replace('/images/', '/masks/')))
        lbl_msk1 = np.array(Image.open(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')))
        
        x0 = 512
        y0 = 512

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        lbl_msk1 = lbl_msk1[y0:y0+crop_size, x0:x0+crop_size]

        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)
        
        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.tensor(img.transpose((2, 0, 1))).float()
        msk = torch.tensor(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk}
        return sample


def validate(model, data_loader):
    dices0 = []

    tp = np.zeros((4,))
    fp = np.zeros((4,))
    fn = np.zeros((4,))
    totalp = np.zeros((4,))

    
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            # msk_loc = sample["msk_loc"].numpy() * 1
            out = model(imgs)

            # msk_pred = msk_loc
            msk_pred = torch.sigmoid(out).cpu().numpy()[:, 0, ...]
            msk_damage_pred = torch.sigmoid(out).cpu().numpy()[:, 1:, ...]
            
            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))
                targ = lbl_msk[j][lbl_msk[j, 0] > 0]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > _thr)
                pred = pred[lbl_msk[j, 0] > 0]
                for c in range(4):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()
                    totalp += (targ == c).sum()

    d0 = np.mean(dices0)

    f1_sc = np.zeros((4,))
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * d0 + 0.7 * f1
    print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))
    return sc


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
            'optimizer' : optimizer.state_dict(),
        }, path.join(models_folder, snapshot_name))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, model, discriminator, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    lossesgan = AverageMeter()

    seg_loss = ComboLoss({'dice': 1, 'focal': 8}, per_image=False).cuda()
    weights_ = torch.tensor([0.001, 0.10,1.5,1.5,1.5])
    ce_loss = nn.CrossEntropyLoss(weight=weights_).cuda()
    gan_loss = nn.BCEWithLogitsLoss().cuda()

    # weights_ = torch.tensor([0.10,1.,1., 1.])
    # ce_loss_avg = nn.CrossEntropyLoss(weight=weights_).cuda()
    # ce_loss = nn.CrossEntropyLoss().cuda()


    iterator = tqdm(train_data_loader)
    # iterator = train_data_loader
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)
        
        #### GENERATOR ###
        model.zero_grad()
        out = model(imgs)

        if (i % 8 == 0):
            #### DISCRIMININATOR ###
            discriminator.zero_grad()

            msks = msks.to(torch.float32)
            true_label = discriminator(msks)
            loss_gan_1 = gan_loss(true_label, valid)
            fake_label = discriminator(out.detach())
            loss_gan_0 = gan_loss(fake_label, fake)
            loss_D = 0.1*(loss_gan_1 + loss_gan_0)/2
            loss_D.backward()
            d_optimizer.step()

        #### GENERATOR ###
        loss_seg = seg_loss(out[:, 0, ...], msks[:, 0, ...])

        msks[:, 0, ...] = 1 - msks[:, 0, ...]
        lbl_msk = torch.argmax(msks, dim=1)
        loss_cls = ce_loss(out, lbl_msk) * 5

        # out_ordinal = torch.argmax(out, dim=1)
        # lbl_msk = lbl_msk.float()
        # loss_ordinal = nn.MSELoss()(out_ordinal, lbl_msk)/5

        gen_label = discriminator(out.detach())
        loss_gan = gan_loss(gen_label, valid)
        loss_G = loss_seg + loss_cls + 0.01*loss_gan
                   
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

        losses.update(loss_G.item(), imgs.size(0))
        losses1.update(loss_D.item(), imgs.size(0)) #loss5
        losses2.update(loss_cls.item(), imgs.size(0))
        lossesgan.update(loss_gan, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f}; loss_D {loss1.val:.4f}; loss_cls {loss2.avg:.4f}; loss_gan {dice.val:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, loss2=losses2, dice=lossesgan))


    scheduler.step(current_epoch)
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; loss2 {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=lossesgan))


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
        # if (random.random() > 0.7) and file_classes[i, 3].max():
        #     train_idxs.append(i)
        #     non_zero_dmg += 1

    train_idxs = np.asarray(train_idxs)
    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size
    print(non_zero_bldg, non_zero_dmg, len(train_idxs), len(val_idxs))
    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    discriminator = Discriminator().cuda()

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=8, shuffle=False, pin_memory=False)

    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    d_optimizer = AdamW(discriminator.parameters(), lr=0.0001, weight_decay=1e-6)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.6)
    
    # snap_to_load = 'res34_loc_{}_1_best'.format(seed)
    '''
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
    del loaded_dict
    del sd
    del checkpoint
'''
    gc.collect()
    torch.cuda.empty_cache()

    model = nn.DataParallel(model).cuda()
    # discriminator = nn.DataParallel(discriminator)

    best_score = 0
    torch.cuda.empty_cache()

    scaler = amp.GradScaler()

    for epoch in range(100):
        train_epoch(epoch, model, discriminator, optimizer, scheduler, train_data_loader)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    torch.cuda.empty_cache()
    print('Time: {:.3f} min'.format(elapsed / 60))

