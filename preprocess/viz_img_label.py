import numpy as np
import argparse
from pathlib import Path
import cv2 as cv
from os.path import join
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-img_dir', help='input annotated directory')
    parser.add_argument('-label_dir', help='output dataset directory')
    parser.add_argument('-out_dir', help='output dataset directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in Path(args.out_dir).glob('*.*'):
        os.remove(str(path))

    img_paths = list([path for path in Path(args.img_dir).glob('*.jpg')])
    for path in tqdm(img_paths):
        mask_path = join(*[args.label_dir, path.name])
        img = cv.imread(str(path))
        mask = cv.imread(str(mask_path))
        plt.clf()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.4)
        plt.savefig(join(*[args.out_dir, path.name]))
