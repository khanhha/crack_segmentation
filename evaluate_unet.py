from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-8) / (y_true.sum() + y_pred.sum() + 1e-8)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    # Intersection = true positives
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    # assert y_true.ndim == 1
    # assert y_pred.ndim == 1 or y_pred.ndim == 2

    # if y_pred.ndim == 2:
    #    y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum()#.to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum()#.to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum()#.to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum()#.to(torch.float32)

    epsilon = 1e-8

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # f1.requires_grad = is_training
    return f1

def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    epsilon = 1e-8

    return tp / (tp + fn + epsilon)

def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    epsilon = 1e-8

    return tp / (tp + fp + epsilon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-ground_truth_dir', type=str,  required=True, help='path where ground truth images are located')
    arg('-pred_dir', type=str, required=True,  help='path with predictions')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    args = parser.parse_args()

    result_f1 = []
    result_dice = []
    result_jaccard = []
    result_precision = []
    result_recall = []

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

        result_f1 += [f1_loss(y_true, y_pred)]
        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
        result_precision += [precision(y_true, y_pred)]
        result_recall += [recall(y_true, y_pred)]


    print('F1 loss = ', np.mean(result_f1), np.std(result_f1))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('Precision = ', np.mean(result_precision), np.std(result_precision))
    print('Recall = ', np.mean(result_recall), np.std(result_recall))
