import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc

def load_unet_vgg16(model_path):
    model = UNet16(pretrained=True)
    checkpoint = torch.load(model_path)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['check_point'])
    else:
        raise Exception('undefind model format')

    model.cuda()
    model.eval()

    return model

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)

    return mask

def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.2
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-out_dir', type=str, help='trained model path')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    #for path in Path(args.out_dir).glob('*.*'):
    #    os.remove(str(path))

    model = load_unet_vgg16(args.model_path)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    for path in Path(args.img_dir).glob('*.*'):
        # if '01-01-01' not in path.stem:
        #       continue

        print(str(path))

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        img_height, img_width, img_channels = img_0.shape

        prob_map = evaluate_img_patch(model, img_0)

        plt.clf()
        plt.subplot(231); disable_axis()
        plt.imshow(img_0)
        plt.subplot(232); disable_axis()
        plt.imshow(prob_map)
        plt.subplot(233); disable_axis()
        plt.imshow(img_0)
        plt.imshow(prob_map, alpha=0.5)

        mask = evaluate_img(model, img_0)

        plt.subplot(234); disable_axis()
        plt.imshow(img_0)
        plt.subplot(235); disable_axis()
        plt.imshow(mask)
        plt.subplot(236); disable_axis()
        plt.imshow(img_0)
        plt.imshow(mask, alpha=0.5)
        #plt.show()

        plt.savefig(join(args.out_dir, f'{path.stem}.jpg'), dpi=500)

        gc.collect()