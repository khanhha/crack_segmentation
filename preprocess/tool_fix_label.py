import numpy as np
import argparse
from pathlib import Path
import cv2 as cv
from os.path import join
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-img_dir', help='input annotated directory')
    parser.add_argument('-label_dir', help='output dataset directory')
    parser.add_argument('-out_img_dir', help='output dataset directory')
    parser.add_argument('-out_mask_dir', help='output dataset directory')
    args = parser.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)

    label_names = set([path.stem for path in Path(args.label_dir).glob('*.png')])
    for path in Path(args.img_dir).glob('*.jpg'):
        if path.stem not in label_names:
            print(f'missing label {path.stem}')
            continue

        if '2815' not in path.stem:
             continue
        img = cv.imread(str(path))
        #lb = np.load(join(*[args.label_dir, f'{path.stem}.npy']))
        lb = cv.imread(join(*[args.label_dir, f'{path.stem}.png']))
        lb = cv.cvtColor(lb, cv.COLOR_BGR2GRAY)
        lb = (lb > 0).astype(np.uint8)*255
        lb = cv.morphologyEx(src=lb,op=cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
        plt.imshow(img)
        plt.imshow(lb, alpha=0.5)
        plt.show()
        cv.imwrite(filename=join(*[args.out_img_dir],  path.name), img=img)
        cv.imwrite(filename=join(*[args.out_mask_dir], path.name), img=lb)