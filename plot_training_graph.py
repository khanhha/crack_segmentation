import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir", type=str, required=True)
    ap.add_argument("-title", type=str, required=True)
    args = ap.parse_args()

    paths = [path for path in Path(args.model_dir).glob('*.pt')]
    paths = sorted(paths)
    epochs = []
    tr_losses = []
    vl_losses = []
    for path in tqdm(paths):
        if 'epoch' not in path.stem:
            continue
        #load the min loss so far
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)
        state = torch.load(path)
        val_los = state['valid_loss']
        train_loss = float(state['train_loss'])
        tr_losses.append(train_loss)
        vl_losses.append(val_los)

    sorted_idxs = np.argsort(epochs)
    tr_losses = [tr_losses[idx] for idx in sorted_idxs]
    vl_losses = [vl_losses[idx] for idx in sorted_idxs]

    print(tr_losses)
    print(vl_losses)

    plt.plot(tr_losses[1:], label='train_loss')
    plt.plot(vl_losses[1:], label='valid_loss')
    plt.title(args.title)
    plt.legend()
    plt.show()





