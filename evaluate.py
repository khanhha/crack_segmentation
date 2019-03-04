import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
from unet import UNet
from unet.unet_transfer import UNetVGG16
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image

def load_unet_vgg16(model_path):
    model = UNetVGG16(pretrained=True)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    return model

def load_unet(model_path):
    model = UNet(n_channels=3, n_classes=1)
    model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-out_dir', type=str, help='trained model path')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_unet_vgg16(args.model_path)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

    for path in Path(join(args.test_dir, 'images')).glob('*.*'):
        print(str(path))
        img_0 = Image.open(str(path))
        img = train_tfms(img_0)
        img = img.unsqueeze(0)
        X = Variable(img).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        plt.clf()
        plt.axis('off')
        plt.subplot(131)
        plt.imshow(np.asarray(img_0, np.uint8))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(mask)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(np.asarray(img_0, np.uint8))
        plt.imshow(mask, alpha=0.3)
        plt.axis('off')
        plt.savefig(join(args.out_dir, path.name), dpi=300)
