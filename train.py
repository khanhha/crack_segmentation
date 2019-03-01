import sys
import os
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt
import argparse
from torchsummary import summary
from data_loader import ImgDataSet

def get_loss(dl, model):
    loss = 0
    for X, y in dl:
        X, y = Variable(X).cuda(), Variable(y).cuda()
        output = model(X)
        loss += F.cross_entropy(output, y).data[0]
    loss = loss / len(dl)
    return loss

from torch.nn.modules.module import _addindent
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='output dataset directory')
    args = parser.parse_args()

    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')
    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]
    print(f'total training images = {len(img_names)}')

    channel_means = (0.20166926, 0.28220195, 0.31729624)
    channel_stds = (0.20769505, 0.18813899, 0.16692209)

    class param:
        img_size = (224, 224)
        bs = 8
        num_workers = 4
        lr = 0.001
        epochs = 5
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
    print(torch_summarize(model))
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
            #print(masks_probs_flat.shape)
            #print(true_masks_flat.shape)
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += float(loss)

            print('{0:.4f} --- loss: {1:.6f}'.format(i * param.bs / N_train, float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), args.model_path)

    # model.eval()
    # for i in range(len(img_names)):
    #     fname = img_names[i]
    #     fpath = os.path.join(DIR_IMG, fname)
    #     img_0 = Image.open(fpath)
    #     img = train_tfms(img_0)
    #     img = img.unsqueeze(0)
    #     X = Variable(img).cuda()  # [N, 1, H, W]
    #     masks_pred = model(X)
    #     mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
    #     print('khanh_3 ',mask.shape)
    #     #mask = cv.resize((mask*255).astype(np.uint8), dsize=(224,224))
    #     print(np.min(mask[:]), np.max(mask[:]))
    #     plt.subplot(131)
    #     plt.imshow(np.asarray(img_0, np.uint8))
    #     plt.subplot(132)
    #     plt.imshow(mask)
    #     plt.subplot(133)
    #     plt.imshow(np.asarray(img_0, np.uint8))
    #     plt.imshow(mask, alpha=0.3)
    #     plt.show()

