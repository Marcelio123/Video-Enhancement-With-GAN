import random
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image


def rand_crop(*args, sz):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(args[0], output_size=sz)
    out = []
    for im in args:
        out.append(TF.crop(im, i, j, h, w))
    return out

def center_crop(*args, sz):
    h, w = sz
    out = []
    for im in args:
        img_w, img_h = im.size  # Accessing image size directly
        
        # Calculate padding
        pad_h = max(h - img_h, 0)
        pad_w = max(w - img_w, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        
        # Perform padding and then crop
        padded_im = Image.new(im.mode, (img_w + pad_w, img_h + pad_h), color='black')
        padded_im.paste(im, (pad_w // 2, pad_h // 2))
        i = (padded_im.size[1] - h) // 2
        j = (padded_im.size[0] - w) // 2
        out.append(TF.crop(padded_im, i, j, h, w))
    return out

def rand_flip(*args, p):
    out = list(args)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.hflip(im)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.vflip(im)
    return out


def rand_reverse(*args, p):
    if random.random() < p:
        return args[::-1]
    else:
        return args