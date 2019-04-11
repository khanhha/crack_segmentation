import torch
from torch import nn
from unet.unet_transfer import UNet16, UNetResNet
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSetJoint, ImgDataSet
import os
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
import albumentations as albu
from albumentations.pytorch import ToTensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device, type ='vgg16'):
    assert type == 'vgg16' or type == 'resnet101'
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    else:
        assert False
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args):

    latest_model_path = find_latest_model_path(args.model_dir)

    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_pred = masks_pred.view(-1)
            target_var = target_var.view(-1)
            loss = criterion(masks_pred, target_var)

            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)

            output = output.view(-1)
            target_var = target_var.view(-1)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

def create_loader(dir, args):
    img_dir = os.path.join(*[dir, 'images'])
    mask_dir = os.path.join(*[dir, 'masks'])
    img_names  = sorted([path.name for path in Path(img_dir).glob('*.jpg')])
    mask_names = sorted([path.name for path in Path(mask_dir).glob('*.jpg')])

    #sanity checking'
    assert len(img_names) == len(mask_names), 'mismatched number of image and masks'
    for img_name, mask_name in zip(img_names, mask_names):
        assert img_name == mask_name, 'mismatched image name vs mask name'

    #join_tfms = albu.Compose([albu.VerticalFlip(), albu.HorizontalFlip(), albu.ShiftScaleRotate()])
    join_tfms = albu.Compose([albu.VerticalFlip(), albu.HorizontalFlip()])
    #img_tfms  = albu.Compose([albu.RandomBrightnessContrast(), albu.RandomGamma(), albu.Normalize(), ToTensor()])
    img_tfms = albu.Compose([albu.Normalize(), ToTensor()])

    mask_tfms = albu.Compose([ToTensor()])

    #dataset = ImgDataSetJoint(img_dir=img_dir, img_fnames=img_names, mask_dir=mask_dir, mask_fnames=mask_names, joint_transform=join_tfms, img_transform=img_tfms, mask_transform=mask_tfms)
    dataset = ImgDataSet(img_dir=img_dir, img_fnames=img_names, mask_dir=mask_dir, mask_fnames=mask_names, img_transform=img_tfms, mask_transform=mask_tfms)

    train_loader = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    return train_loader

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('-data_dir', type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='resnet101', help='vgg16 or resnet101')

    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size', default=4, type=int, help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(device, args.model_type)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # crack_weight = 0.4*calc_crack_pixel_weight(DIR_MASK)
    # print(f'positive weight: {crack_weight}')
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([crack_weight]).to('cuda'))
    criterion = nn.BCEWithLogitsLoss().to('cuda')

    train_dir = os.path.join(*[args.data_dir, 'train'])
    valid_dir = os.path.join(*[args.data_dir, 'valid'])

    train_loader = create_loader(train_dir, args)
    valid_loader = create_loader(valid_dir, args)

    model.cuda()
    train(train_loader, valid_loader, model, criterion, optimizer, validate, args)

if __name__ == '__main__':
    main()

