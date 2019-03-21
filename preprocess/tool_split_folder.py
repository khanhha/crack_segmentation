import numpy as np
import argparse
from pathlib import Path
import shutil
import os
import scipy.io as io
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os.path import join

if __name__ == '__main__':
    name = 'merged_google_crack_images'
    in_dir = f'/home/khanhhh/data_1/courses/practical_project_1/codes/dataset/google_images/{name}'
    out_dir = '/home/khanhhh/data_1/courses/practical_project_1/codes/dataset/google_images/'
    paths = [path for path in Path(in_dir).glob('*.jpg')]

    chunk_size = 150
    for i in range(0, len(paths), chunk_size):
        i_n = min(i + chunk_size, len(paths))
        paths_subset = paths[i:i_n]
        out_dir_1 = f'{out_dir}/{name}_{i}'
        os.makedirs(out_dir_1, exist_ok=True)
        for path in paths_subset:
            shutil.copy(src=str(path), dst=f'{out_dir_1}/{path.name}')

