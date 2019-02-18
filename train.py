import sys
import os
from optparse import OptionParser
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt

class ImgDataSet(Dataset):
    def __init__(self, img_dir, img_fnames, img_transform, mask_dir, mask_fnames, mask_transform):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)
            #print('image shape', img.shape)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)
        #print('khanh1', np.min(test[:]), np.max(test[:]))
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            #print('mask shape', mask.shape)
            #print('khanh2', np.min(test[:]), np.max(test[:]))

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)


def get_loss(dl, model):
    loss = 0
    for X, y in dl:
        X, y = Variable(X).cuda(), Variable(y).cuda()
        output = model(X)
        loss += F.cross_entropy(output, y).data[0]
    loss = loss / len(dl)
    return loss

if __name__ == '__main__':
    DIR_CHECKPOINT = '/home/khanhhh/data_1/courses/practical_project_1/codes/dataset/merged_segmentation_tmp/'
    ROOT_DIR = '/home/khanhhh/data_1/courses/practical_project_1/codes/dataset/merged_segmentation'
    DIR_IMG = f'{ROOT_DIR}/images/'
    DIR_MASK = f'{ROOT_DIR}/masks/'
    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

    channel_means = (0.20166926, 0.28220195, 0.31729624)
    channel_stds = (0.20769505, 0.18813899, 0.16692209)

    class param:
        img_size = (224, 224)
        bs = 8
        num_workers = 4
        lr = 0.001
        epochs = 30
        log_interval = 70  # less then len(train_dl)

    train_tfms = transforms.Compose([#transforms.Resize(param.img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([#transforms.Resize(param.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([#transforms.Resize(param.img_size,interpolation=Image.NEAREST),
                                    transforms.ToTensor()])

    train_dl = DataLoader(ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms),
                          param.bs, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=param.num_workers)


    model = UNet(n_channels=3, n_classes=1)
    model.cuda()
    cudnn.benchmark = True # faster convolutions, but more memory

    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    criterion = nn.BCELoss()

    N_train = len(train_dl)
    model.train()
    for epoch in range(param.epochs):

        epoch_loss = 0

        for i, (X,y) in enumerate(train_dl):
            X = Variable(X).cuda()  # [N, 1, H, W]
            y = Variable(y).cuda()  # [N, H, W] with class indices (0, 1)

            masks_pred = model(X)
            masks_probs_flat = masks_pred.view(-1)

            # print(masks_pred[0])
            # plt.imshow(np.asarray(masks_pred[0], dtype=np.float))
            # plt.show()

            true_masks_flat = y.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += float(loss)

            print('{0:.4f} --- loss: {1:.6f}'.format(i * param.bs / N_train, float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    for i in range(len(img_names)):
        fname = img_names[i]
        fpath = os.path.join(DIR_IMG, fname)
        img_0 = Image.open(fpath)
        img = train_tfms(img_0)
        img = img.unsqueeze(0)
        X = Variable(img).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        print('khanh_3 ',mask.shape)
        mask = cv.resize((mask*255).astype(np.uint8), dsize=(480,320))
        print(np.min(mask[:]), np.max(mask[:]))
        plt.imshow(np.asarray(img_0, np.uint8))
        plt.imshow(mask, alpha=0.5)
        plt.show()

