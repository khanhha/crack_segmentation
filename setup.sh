#!/bin/bash
mkdir CRACK500
mkdir CRACK500/test
mkdir CRACK500/test/images
mkdir CRACK500/test/masks
mkdir CRACK500/train
mkdir CRACK500/train/images
mkdir CRACK500/train/masks

mkdir CFD
mkdir CFD/test
mkdir CFD/test/images
mkdir CFD/test/masks
mkdir CFD/train
mkdir CFD/train/images
mkdir CFD/train/masks

unzip crack_segmentation_dataset.zip

mv crack_segmentation_dataset/train/images/CRACK500* CRACK500/train/images
mv crack_segmentation_dataset/test/images/CRACK500* CRACK500/test/images
mv crack_segmentation_dataset/train/masks/CRACK500* CRACK500/train/masks
mv crack_segmentation_dataset/test/masks/CRACK500* CRACK500/test/masks

mv crack_segmentation_dataset/train/images/CFD* CFD/train/images
mv crack_segmentation_dataset/test/images/CFD* CFD/test/images
mv crack_segmentation_dataset/train/masks/CFD* CFD/train/masks
mv crack_segmentation_dataset/test/masks/CFD* CFD/test/masks

mkdir Experiment/CRACK500/model
mkdir Experiment/CRACK500/model/first
mkdir Experiment/CRACK500/model/second
mkdir Experiment/CRACK500/model/third
mkdir Experiment/CRACK500/test_results/first
mkdir Experiment/CRACK500/test_results/second
mkdir Experiment/CRACK500/test_results/third

mkdir Experiment/CFD/model
mkdir Experiment/CFD/model/first
mkdir Experiment/CFD/model/second
mkdir Experiment/CFD/model/third
mkdir Experiment/CFD/test_results/first
mkdir Experiment/CFD/test_results/second
mkdir Experiment/CFD/test_results/third

conda create --name crack
conda activate crack
conda install -c anaconda pytorch-gpu -y
conda install -c conda-forge opencv -y
conda install matplotlib scipy numpy tqdm pillow -y
conda install numba -y
pip install torchvision
pip install torchgeometry