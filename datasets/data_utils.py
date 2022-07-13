import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    print(imgs.mean(), imgs.std())

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]

    return imgs, labels


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            with_random_resize=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_resize = with_random_resize
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
    def transform(self, imgs, labels, to_tensor=True, split='', patch=None):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        
        if self.img_size is None:
            self.img_size = None

        if split=='train':
            x0 = random.randint(0, imgs[0].size[1] - self.img_size)
            y0 = random.randint(0, imgs[0].size[0] - self.img_size)
        else:
            if patch:
                x0, y0 = 256*(patch//4), 256*(patch%4)
            else:
                x0, y0 = (256,256)
            # if random.random() > 0.5:
            #     x0, y0 = (256,256)
            # else:
            #     x0, y0 = (768,768)
                
        imgs = [TF.to_pil_image(img) for img in imgs]

        if self.img_size < imgs[0].size[0]//2:
            imgs = [Image.fromarray(np.array(img)[y0:y0+self.img_size, x0:x0+self.img_size, :]) for img in imgs]
            labels = [Image.fromarray(np.array(img)[y0:y0+self.img_size, x0:x0+self.img_size]) for img in labels]
        else:
            labels = [Image.fromarray(np.array(img)) for img in labels]

    
        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels

class CDDataAugmentation_xBD:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
    
    def transform(self, imgs, labels, to_tensor=True, is_train=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if is_train==True:
            x0 = random.randint(0, imgs[0].size[1] - self.img_size)
            y0 = random.randint(0, imgs[0].size[0] - self.img_size)
        else:
            x0, y0 = (256,256)

        imgs = [Image.fromarray(np.array(img)[y0:y0+self.img_size, x0:x0+self.img_size, :]) for img in imgs]
        labels = [Image.fromarray(np.array(img)[y0:y0+self.img_size, x0:x0+self.img_size]) for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        img = imgs[0]
        if random.random() > 0.98:
            if random.random() > 0.985:
                img = clahe(img)
            elif random.random() > 0.985:
                img = gauss_noise(img)
            elif random.random() > 0.985:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.98:
            if random.random() > 0.985:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = contrast(img, 0.9 + random.random() * 0.2)
        imgs[0] = img
        
        img = imgs[1]
        if random.random() > 0.98:
            if random.random() > 0.985:
                img = clahe(img)
            elif random.random() > 0.985:
                img = gauss_noise(img)
            elif random.random() > 0.985:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.98:
            if random.random() > 0.985:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.985:
                img = contrast(img, 0.9 + random.random() * 0.2)
        imgs[1] = img


        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
