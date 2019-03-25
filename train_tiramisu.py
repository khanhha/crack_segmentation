import os
import sys
import math
import string
import random
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
from utils import AverageMeter
from pathlib import Path
import densenet.tiramisu as tiramisu
from data_loader import ImgDataSetJoint
from torch.utils.data import DataLoader, Dataset, random_split
from joint_transforms import JointRandomSizedCrop
import argparse
import tqdm

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)

    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)

    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(err,5)

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args):
    # switch to train mode
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if Path(best_model_path).exists():
        state = torch.load(args.model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}'.format(epoch))
    else:
        epoch = 0

    valid_losses = []

    min_val_los = 9999

    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        losses = AverageMeter()

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()
            masks_pred = model(input_var)
            #assert (masks_probs_flat >= 0. & masks_probs_flat <= 1.).all()
            masks_pred = masks_pred.view(-1)
            target_var  = target_var.view(-1)
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
        print(f'\tvalid_loss = {valid_loss:.5f}\n')

        tq.close()

        #save the model of the current epoch
        epoc_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoc_model_path)

        #save the best model so far
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
        #tq = tqdm.tqdm(total=(len(val_loader) * args.batch_size))
        #tq.set_description(f'Validation ')

        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

            #tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            #tq.update(args.batch_size)

        #tq.close()
    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=2, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')

    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')

    LR = 1e-4
    LR_DECAY = 0.995
    DECAY_EVERY_N_EPOCHS = 1
    N_EPOCHS = 2

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

    print(f'total training images = {len(img_names)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    train_joint_trans = JointRandomSizedCrop(size=224)
    train_mask_trans = transforms.ToTensor()

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    val_joint_trans = JointRandomSizedCrop(size=224)
    val_mask_trans = transforms.ToTensor()

    dataset = ImgDataSetJoint(img_dir=DIR_IMG, img_fnames=img_names, joint_transform=train_joint_trans, mask_dir=DIR_MASK, mask_fnames=mask_names, img_transform=train_tfms, mask_transform=train_mask_trans)
    train_size = int(0.85*len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model = tiramisu.FCDenseNet67(n_classes=1).cuda()
    model.apply(weights_init)
    model.cuda()
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #criterion = nn.NLLLoss2d().cuda()
    criterion = nn.BCEWithLogitsLoss().to('cuda')

    train(train_loader, valid_loader, model, criterion, optimizer, validate, args)
