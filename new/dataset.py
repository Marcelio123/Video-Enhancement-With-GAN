import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import vfitransform as vt
import os.path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def get_flow(im1, im2, flag=True):
    im1 = np.array(im1)
    im2 = np.array(im2)
    if flag:
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
    else:
        im1 = im1.astype(float)
        im2 = im2.astype(float)
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in
    
    

class Vimeo90k_septuplet(data.Dataset):
    def __init__(self, db_dir, train=False, upscale_factor=4, transform=None, crop_sz=(256,256), augment_s=False, augment_t=False):
        seq_dir = join(db_dir, 'sequences')
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'fast_testset.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index], "im1.png"))
        rawFrame2 = Image.open(join(self.seq_path_list[index], "im2.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index], "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index], "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index], "im5.png"))
        rawFrame6 = Image.open(join(self.seq_path_list[index], "im6.png"))
        rawFrame7 = Image.open(join(self.seq_path_list[index], "im7.png"))

        if self.crop_sz is not None:
            # rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_crop(
            #     rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, sz=self.crop_sz
            # )
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.center_crop(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, sz=self.crop_sz
            )

        if self.augment_s:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_flip(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )

        if self.augment_t:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_reverse(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )

        rawFrames = [rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7]
        targets = [0, 2, 3, 4, 6]

        dataForVSR = []

        for i in targets:
            target = rawFrames[i]
            input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)
            neigbor = [rawFrames[j] for j in range(len(rawFrames)) if j != i]
            neigbor = [j.resize((int(j.size[0]/self.upscale_factor),int(j.size[1]/self.upscale_factor)), Image.BICUBIC) for j in neigbor]
            flow = [get_flow(input,j) for j in neigbor]
            bicubic = rescale_img(input, self.upscale_factor)

            if self.transform:
                target = self.transform(target)
                input = self.transform(input)
                bicubic = self.transform(bicubic)
                neigbor = [self.transform(j) for j in neigbor]
                flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

            dataForVSR.append([input, target, neigbor, flow, bicubic])
        
        if self.transform:
            rawFrames = [self.transform(rawFrame) for rawFrame in rawFrames]

        return dataForVSR, rawFrames
        

    def __len__(self):
        return len(self.seq_path_list)

class Vimeo90k_septuplet2(data.Dataset):
    def __init__(self, db_dir, train=False, upscale_factor=4, transform=None, crop_sz=(256,256), augment_s=False, augment_t=False):
        seq_dir = join(db_dir, 'sequences')
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'fast_testset.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index], "im1.png"))
        rawFrame2 = Image.open(join(self.seq_path_list[index], "im2.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index], "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index], "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index], "im5.png"))
        rawFrame6 = Image.open(join(self.seq_path_list[index], "im6.png"))
        rawFrame7 = Image.open(join(self.seq_path_list[index], "im7.png"))

        if self.crop_sz is not None:
            # rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_crop(
            #     rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, sz=self.crop_sz
            # )
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.center_crop(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, sz=self.crop_sz
            )

        if self.augment_s:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_flip(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )

        if self.augment_t:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_reverse(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )
        
        rawFrames = [rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7]
        lrFrames = [j.resize((int(j.size[0]/self.upscale_factor),int(j.size[1]/self.upscale_factor)), Image.BICUBIC) for j in rawFrames]
        # bicubics = [rescale_img(j, self.upscale_factor) for j in lrFrames]
        if self.transform:
            rawFrames = [self.transform(j) for j in rawFrames]
            lrFrames = [self.transform(j) for j in lrFrames]
            # bicubics = [self.transform(j) for j in bicubics]
        return rawFrames, lrFrames

    def __len__(self):
        return len(self.seq_path_list)

class Vimeo90k_septuplet_fi(data.Dataset):
    def __init__(self, db_dir, train=False, upscale_factor=4, transform=None, crop_sz=(256,256), augment_s=False, augment_t=False):
        seq_dir = join(db_dir, 'sequences')
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'fast_testset.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index], "im1.png"))
        rawFrame2 = Image.open(join(self.seq_path_list[index], "im2.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index], "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index], "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index], "im5.png"))
        rawFrame6 = Image.open(join(self.seq_path_list[index], "im6.png"))
        rawFrame7 = Image.open(join(self.seq_path_list[index], "im7.png"))

        if self.crop_sz is not None:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_crop(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, sz=self.crop_sz
            )

        if self.augment_s:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_flip(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )

        if self.augment_t:
            rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7 = vt.rand_reverse(
                rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7, p=0.5
            )

        rawFrames = [rawFrame1, rawFrame2, rawFrame3, rawFrame4, rawFrame5, rawFrame6, rawFrame7]
        
        if self.transform:
            rawFrames = [self.transform(rawFrame) for rawFrame in rawFrames]

        return rawFrames[0], rawFrames[1], rawFrames[2], rawFrames[3], rawFrames[4], rawFrames[5], rawFrames[6], 
        

    def __len__(self):
        return len(self.seq_path_list)