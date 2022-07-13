import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
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
import torch.cuda.amp as amp
from adamw import AdamW
from losses import dice_round, ComboLoss

from tqdm import tqdm
import timeit
from utils import *
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()

from zoo.models import BASE_Transformer, Res34_Unet_Double
from zoo.model_transformer_encoding import BASE_Transformer_UNet, Discriminator
from PIL import Image

model = "TUNet"
device = ('cuda' if torch.cuda.is_available() else 'cpu')

if model == "TUNet":
    print("UNet Transformer")
    model = BASE_Transformer_UNet(input_nc=3, output_nc=4, token_len=4, resnet_stages_num=4,
                               with_pos='learned', with_decoder_pos='learned', enc_depth=1, dec_depth=8).to(device)
    snapshot_name = 'BASE_UNet_Transformer_img1024_lossOrig_idadata_xBD_4class'
    print("snapshot_name ", snapshot_name, "with seg and cls headers and ce loss only on building")
    snap_to_load = 'BASE_UNet_Transformer_img1024_lossOrig_alldata'

elif model == "BiT":
    print("BiT ....")
    model = BASE_Transformer(input_nc=3, output_nc=4, token_len=4, resnet_stages_num=4,
                              with_pos='learned', enc_depth=1, dec_depth=8).to(device)
    snapshot_name = 'BiT_lossv2'
    print("snapshot_name ", snapshot_name)
    print("Loss only building patch lr:0.001 Seg weights: loss_seg = loss0 ")
    print("CE weights_ = torch.tensor([0.001,0.10,1.5,1.0,1.5])")
    print("reduced upsampling of images 1 and 3")
    snap_to_load = 'res34_loc_0_1_best'

else:
    print("Siamese ResNEt")
    model = Res34_Unet_Double().to(device)
    snapshot_name = 'Res34_Unet_Double_img1024_Ida'
    snap_to_load = 'res34_cls2_0_tuned_best'


train_dirs = ['../data/IdaBD']#, '/data/train']#
models_folder = 'weights'

input_shape = (1024,1024)
crop_size = 1024
_thr = 0.3
cudnn.benchmark = True
batch_size = 1
val_batch_size = 1


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if ('_pre_disaster.png' in f) and (('hurricane-michael' in f) or ('AOI' in f)):
            path_ = path.join(d, 'images', f)
            path_ = path_.replace('/', '//')
            all_files.append(path_)


def normalize_xbd(img):
    img = (img - [87.4, 96.4,  74.7])/([41.8, 37.8, 37.9])
    img = img*([44.7, 38.7, 33.8]) + [75.1,  74.3, 56.4]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TrainData(Dataset):
    def __init__(self, train_idxs, all_nonzero_files):
        super().__init__()
        self.train_idxs = train_idxs
        self.all_files = all_nonzero_files

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = self.all_files[_idx]
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
        '''        
        
        if 'hurricane' in fn:
            img1 = normalize_xbd(img1)
            img2 = normalize_xbd(img2)

        if random.random() > 0.3:
            imgs = [img1, img2]
            labels = [msk0, lbl_msk1]
            imgs = [TF.to_pil_image(img) for img in imgs]
            labels = [TF.to_pil_image(img) for img in labels]

            if random.random() > 0.5:
                imgs = [TF.hflip(img) for img in imgs]
                labels = [TF.hflip(img) for img in labels]

            if random.random() > 0.5:
                imgs = [TF.vflip(img) for img in imgs]
                labels = [TF.vflip(img) for img in labels]
            
            if random.random() > 0.5:
                x = random.randint(0, 200)
                y = random.randint(0, 200)
                imgs = [TF.resized_crop(img, x, y, crop_size-x, crop_size-y, (crop_size,crop_size)) for img in imgs]
                labels = [TF.resized_crop(img, x, y, crop_size-x, crop_size-y, (crop_size,crop_size)) for img in labels]

            if random.random() > 0.7:
                imgs = [transforms.ColorJitter(brightness=[0.95,1.05], contrast=[0.9,1.1], saturation=[0.9,1.1])(img) for img in imgs]
            
            msk0, lbl_msk1 = np.array(labels[0]), np.array(labels[1])
            img1, img2 = np.array(imgs[0]), np.array(imgs[1])
        '''
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        #msk4 = np.zeros_like(lbl_msk1)
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk3[lbl_msk1 == 4] = 255
        msk1[lbl_msk1 == 1] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        #msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3], axis=2)
        msk = (msk > 127)

        msk[..., 0] = False
        '''msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 3] = dilation(msk[..., 3], square(5))
        msk[..., 4] = dilation(msk[..., 4], square(5))'''
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = True
        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)

        img = np.concatenate([img1, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.tensor(img.transpose((2, 0, 1))).float()
        msk = torch.tensor(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs, all_nonzero_files):
        super().__init__()
        self.image_idxs = image_idxs
        self.all_files = all_nonzero_files

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = self.all_files[_idx]

        img = np.array(Image.open(fn))
        img2 = np.array(Image.open(fn.replace('_pre_disaster', '_post_disaster')))
        # msk_loc = cv2.imread(path.join(loc_folder, '{0}.png'.format(fn.split('/')[-1].replace('.png', '_part1.png'))), cv2.IMREAD_UNCHANGED) > (0.3*255)

        msk0 = np.array(Image.open(fn.replace('/images/', '/masks/')))
        lbl_msk1 = np.array(Image.open(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')))
        
        '''x0 = 512
        y0 = 512

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        lbl_msk1 = lbl_msk1[y0:y0+crop_size, x0:x0+crop_size]
'''
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        #msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk3[lbl_msk1 == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        #msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = msk[..., 1:].argmax(axis=2)
        
        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.tensor(img.transpose((2, 0, 1))).float()
        msk = torch.tensor(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk}
        return sample


def validate(model, data_loader, best_score):
    dices0 = []

    tp = np.zeros((3,))
    fp = np.zeros((3,))
    fn = np.zeros((3,))
    totalp = np.zeros((3,))
    
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            
            if (best_score == 10000) and (i > len(data_loader)//3):
                continue
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].to(device)
            out = model(imgs)

            msk_pred = torch.sigmoid(out).cpu().numpy()[:, 0, ...]
            msk_damage_pred = torch.sigmoid(out).cpu().numpy()[:, 1:, ...]

            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))
                targ = lbl_msk[j][lbl_msk[j, 0] > 0]
                pred = msk_damage_pred[j].argmax(axis=0)
                pred = pred * (msk_pred[j] > _thr)
                pred = pred[lbl_msk[j, 0] > 0]
                for c in range(3):
                    tp[c] += np.logical_and(pred == c, targ == c).sum()
                    fn[c] += np.logical_and(pred != c, targ == c).sum()
                    fp[c] += np.logical_and(pred == c, targ != c).sum()
                    totalp += (targ == c).sum()

    d0 = np.mean(dices0)

    f1_sc = np.zeros((3,))
    for c in range(3):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
    f1 = 3 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * d0 + 0.7 * f1
    if best_score==10000:
        print("Train Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2]))
    else:
        print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2]))
    return sc


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val, best_score=best_score)
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


def train_epoch(current_epoch, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    lossesgan = AverageMeter()

    seg_loss = ComboLoss({'dice': 1, 'focal': 8}, per_image=False).to(device)
    weights_ = torch.tensor([0.1, 0.5, 1.5, 1.5])
    ce_loss = nn.CrossEntropyLoss(weight=weights_).to(device)
    iterator = tqdm(train_data_loader)

    model = model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].to(device)
        msks = sample["msk"].to(device)
    
        model.zero_grad()
        out = model(imgs)
        print(out.shape, msks.shape)
        loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
        loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
        loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
        loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
        loss_seg = 0.1 * loss0 + 0.8 * loss1 + 2 * loss2 + 8 * loss3 #+ 10 * loss4

        msks[:, 0, ...] = 1 - msks[:, 0, ...]
        lbl_msk = torch.argmax(msks, dim=1)
        loss_cls = ce_loss(out, lbl_msk) * 5

        loss = loss_seg + loss_cls

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        losses1.update(loss1.item(), imgs.size(0))
        losses2.update(loss2.item(), imgs.size(0))
        lossesgan.update(loss3.item(), imgs.size(0))

        iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f}; loss_2 {loss1.val:.4f}; loss_3 {loss2.val:.4f}; loss_4 {dice.val:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, loss2=losses2, dice=lossesgan))

    scheduler.step(current_epoch)
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; loss2 {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=lossesgan))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    seed = 0 

    xbd_files = []
    aoi_files = []
    for fn in tqdm(all_files):
        msk = np.array(Image.open(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster')))
        msk = msk[msk > 0]
        if msk.sum() > 500:
            if 'AOI' in fn:
                aoi_files.append(fn)
            else:
                xbd_files.append(fn)

    train_idxs, val_idxs = train_test_split(np.arange(len(aoi_files)), test_size=0.15, random_state=seed)

    np.random.seed(seed + 321)
    random.seed(seed + 321)

    val_data = ValData(val_idxs, aoi_files) 
    train_aoi = [aoi_files[x] for x in train_idxs]

    print(len(xbd_files))
    xbd_files.extend(train_aoi)
    all_files = xbd_files
    train_idxs = range(len(all_files))
    train_data = TrainData(train_idxs, all_files)
    print(len(train_aoi), len(all_files))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size, num_workers=8, shuffle=False, pin_memory=False)

    '''
    if os.path.exists(path.join(models_folder, snap_to_load)):
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            k_ = 'module.'+ k
            if k_ in loaded_dict and sd[k].size() == loaded_dict[k_].size():
                sd[k] = loaded_dict[k_]
                sd[k].requires_grad = False
            else:
                print(k_, k, "failure")
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        del loaded_dict
        del sd
        del checkpoint

    gc.collect()
    torch.cuda.empty_cache()'''

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-7)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.6)

    if device == 'cuda':
        model = nn.DataParallel(model).to(device)

    best_score = 0
    torch.cuda.empty_cache()
    scaler = amp.GradScaler()

    for epoch in range(100):
        train_epoch(epoch, model, optimizer, scheduler, train_data_loader)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    torch.cuda.empty_cache()
    print('Time: {:.3f} min'.format(elapsed / 60))
    
