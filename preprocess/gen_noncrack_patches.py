import argparse
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ColorJitter
from skimage import io
import PIL.Image as Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-out_dir', type=str, help='output dataset directory')
    args = parser.parse_args()

    out_img_dir = os.path.join(*[args.out_dir, 'images'])
    out_mask_dir = os.path.join(*[args.out_dir, 'masks'])
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    for path in Path(out_img_dir).glob('*.*'):
        os.remove(str(path))
    for path in Path(out_mask_dir).glob('*.*'):
        os.remove(str(path))

    dir_id = Path(args.img_dir).stem

    mask = np.zeros(shape=(448,448), dtype=np.uint8)
    for idx, path in enumerate(Path(args.img_dir).glob('*.*')):
        #img = cv.imread(str(path))
        img = Image.open(str(path))
        n_sample = int(3*(img.size[0] / 224))

        aug = Compose([RandomResizedCrop(size=448, scale=(0.5,1.0)), RandomVerticalFlip(), RandomHorizontalFlip(), ColorJitter() ])

        for i in range(n_sample):
            patch = np.asarray(aug(img))
            out_img_path = os.path.join(*[out_img_dir, f'noncrack_{dir_id}_{idx}_{i}.jpg'])
            cv.imwrite(filename=out_img_path, img=patch)
            out_mask_path = os.path.join(*[out_mask_dir, f'noncrack_{dir_id}_{idx}_{i}.jpg'])
            cv.imwrite(filename=out_mask_path, img=mask)
