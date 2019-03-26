from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-ground_truth_dir', type=str,  required=True, help='path where ground truth images are located')
    arg('-pred_dir', type=str, required=True,  help='path with predictions')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    for file_name in tqdm(paths):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = Path(args.pred_dir) / file_name.name
        if not pred_file_name.exists():
            print(f'missing prediction for file {file_name.name}')
            continue

        pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)
        y_pred = pred_image

        # print(y_true.max(), y_true.min())
        # plt.subplot(131)
        # plt.imshow(y_true)
        # plt.subplot(132)
        # plt.imshow(y_pred)
        # plt.subplot(133)
        # plt.imshow(y_true)
        # plt.imshow(y_pred, alpha=0.5)
        # plt.show()

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))