import numpy as np
import argparse
from pathlib import Path
import shutil
import os
import scipy.io as io
import cv2 as cv
import matplotlib.pyplot as plt

def copy_forest(in_dir, out_dir):

    forest_dir = os.path.join(in_dir, 'forest')

    mask_names = set()
    for path in Path(os.path.join(forest_dir,'groundTruth')).glob('*.mat'):
        mat = io.loadmat(str(path))
        mask = mat['groundTruth']['Segmentation'][0,0]
        mask = (mask == 2).astype(np.uint8)*255
        mask_names.add(path.stem)
        out_path = f'{out_dir}/masks/forest_{path.stem}.jpg'
        #print(f'output mask to file {out_path}')
        cv.imwrite(filename=out_path, img = mask)

    for path in Path(os.path.join(forest_dir,'image')).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        out_path = f'{out_dir}/images/forest_{path.name}'
        #git commit -m "first commit"print(f'output image to file {out_path}')
        shutil.copy(src=str(path), dst=out_path)

def copy_cracktree200(in_dir, out_dir):
    id = 'cracktree200'
    root_dir = os.path.join(in_dir, id)
    img_dir = os.path.join(root_dir,'cracktree200rgb')
    mask_dir = os.path.join(root_dir,'cracktree200_gt')

    mask_names = set()
    for path in Path(mask_dir).glob('*.png'):
        mask = cv.imread(str(path))
        mask_names.add(path.stem)
        outpath = os.path.join(*[out_dir, 'masks', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = mask)

    for path in Path(img_dir).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        img = cv.imread(str(path))
        outpath = os.path.join(*[out_dir, 'images', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = img)

def copy_GAPS384(in_dir, out_dir):
    id = 'GAPS384'
    root_dir = os.path.join(in_dir, id)
    img_dir = os.path.join(root_dir,'croppedimg')
    mask_dir = os.path.join(root_dir,'croppedgt')

    mask_names = set()
    for path in Path(mask_dir).glob('*.png'):
        mask = cv.imread(str(path))
        mask_names.add(path.stem)
        outpath = os.path.join(*[out_dir, 'masks', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = mask)

    for path in Path(img_dir).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        img = cv.imread(str(path))
        outpath = os.path.join(*[out_dir, 'images', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',type=str, help='input dataset directory')
    parser.add_argument('-out_dir', type=str, help='output dataset directory')
    args = parser.parse_args()
    print(args)

    os.makedirs(f'{args.out_dir}/images/', exist_ok=True)
    os.makedirs(f'{args.out_dir}/masks/', exist_ok=True)

    copy_forest(args.in_dir, args.out_dir)
    copy_cracktree200(args.in_dir, args.out_dir)
    copy_GAPS384(args.in_dir, args.out_dir)
