import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt


from torchvision import transforms
import torchvision.transforms.functional as F
import random

#https://github.com/pytorch/vision/releases/tag/v0.2.0
def my_segmentation_transform(input, target):
	i, j, h, w = transforms.RandomCrop.get_params(input, (448, 448))
	input = F.crop(input, i, j, h, w)
	target = F.crop(target, i, j, h, w)
	if random.random() > 0.5:
		input = F.hflip(input)
		target = F.hflip(target)
	F.to_tensor(input), F.to_tensor(target)
	return input, target

class CombinedCrackDataset(Dataset):
    def __init__(self, img_dir, img_fnames, img_transform, mask_dir, mask_fnames, mask_transform, function='train'):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
##        print("len(self.mask_fnames): ", len(self.mask_fnames))
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)

    def __getitem__(self, i):
        fname = self.img_fnames[i]
##        print("fname: ", fname, " \ti: ", i)
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
   

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath).convert('L')

        
        ## DATA AUGMENTATION ##

        # random resize        
        ww,hh = 544,384
        factor = 1.17# + random.random()
        img = img.resize((int(factor*ww),int(factor*hh)), Image.ANTIALIAS)
        mask = mask.resize((int(factor*ww),int(factor*hh)), Image.ANTIALIAS) 

        if False:
          # random padding
          factor = random.random()
          desired_size = int(ww * (2.17 + 2*factor)), int(hh * (2.17 + 2*factor))

          new_img = Image.new("RGB", (desired_size[0], desired_size[1]))
          new_mask= Image.new("L", (desired_size[0], desired_size[1]))

          new_img.paste(img, ((desired_size[0]-img.size[0])//2,
                      (desired_size[1]-img.size[1])//2))    
          new_mask.paste(mask, ((desired_size[0]-img.size[0])//2,
                      (desired_size[1]-img.size[1])//2))    

          img = new_img
          mask = new_mask

       

        # crop image
        i, j, h, w = transforms.RandomCrop.get_params(img, (448, 448))
#        i, j, h, w = transforms.RandomResizedCrop.get_params(img, (448, 448))
##        print(i,j,h,w)
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if False:
          # horizontal flip
          if random.random() > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
          
          # vertical flip
          if random.random() > 0.5:
            img = F.vflip(img)
            mask = F.vflip(mask)

          # rotation
          if random.random() > 0.5:
            angle = random.randint(-180, 180)
            img = F.rotate(img, angle)
            mask = F.rotate(mask, angle)
        




        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)
            #print('image shape', img.shape)

        #print('khanh1', np.min(test[:]), np.max(test[:]))
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            #print('mask shape', mask.shape)
            #print('khanh2', np.min(test[:]), np.max(test[:]))




##        print("img.shape: ", img.shape)
##        print("mask.shape: ", mask.shape)
        if True: 
          print("img.max() ", img.max())
          print("img.mean() ", img.mean())
          print("img.min() ", img.min())
          print("mask.max() ", mask.max())
          print("mask.mean() ", mask.mean())
          print("mask.min() ", mask.min())

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)
