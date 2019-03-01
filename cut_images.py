import argparse
import os
from os.path import join
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',type=str, help='input dataset directory')
    parser.add_argument('-out_dir', type=str, help='output dataset directory')
    args = parser.parse_args()

    out_dir = join(*[args.out_dir, 'images'])
    os.makedirs(out_dir, exist_ok=True)

    size = (448,448)
    for path in Path(args.in_dir).glob('*.jpg'):
        img = cv.imread(str(path))
        min_size = min(img.shape[0], img.shape[1])
        for i in range(0,img.shape[0], min_size):
            for j in range(0,img.shape[1], min_size):
                if i+min_size > img.shape[0] or j+min_size > img.shape[1]:
                    continue
                sub_img = img[i:i+min_size, j:j+min_size]
                sub_img = cv.resize(sub_img, size)
                #plt.imshow(sub_img)
                #plt.show()
                cv.imwrite(filename=join(*[out_dir, f'{path.stem}_{i}_{j}.jpg']), img=sub_img)