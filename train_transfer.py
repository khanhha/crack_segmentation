import torch
from torch import nn
from unet.unet_transfer import UNet16
from pathlib import Path
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSet
import os
import argparse

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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_model(device):
    model = UNet16(pretrained=True)
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        #target = target.cuda(async=True)
        input_var  = Variable(input).cuda()
        target_var = Variable(target).cuda()

        masks_pred = model(input_var)

        masks_probs_flat = masks_pred.view(-1)
        true_masks_flat  = target_var.view(-1)

        #assert (masks_probs_flat >= 0. & masks_probs_flat <= 1.).all()
        loss = criterion(masks_probs_flat, true_masks_flat)

        if i % args.print_freq == 0:
            print('{0:.4f} --- loss: {1:.6f}'.format(i, float(loss)))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, args):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

    print(f'valid loss = {losses.avg}')

    return losses.avg

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-p', '--print-freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='output dataset directory')

    args = parser.parse_args()

    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

    print(f'total training images = {len(img_names)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    class param:
        bs = 4
        num_workers = 4
        lr = 0.001
        epochs = 5
        log_interval = 70  # less then len(train_dl)


    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])


    model = get_model(device)

    print(model)
    exit()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    # criterion = nn.BCELoss()

    dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    train_size = int(0.85*len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, param.bs, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=param.num_workers)
    valid_loader = DataLoader(train_dataset, param.bs, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=param.num_workers)

    model.cuda()

    min_loss = 999
    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, args)

        loss = validate(valid_loader, model, criterion, args)

        is_best = loss < min_loss
        min_loss = min(min_loss, loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch':  'vgg16',
            'state_dict': model.state_dict(),
            'best_loss': min_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=args.model_path)