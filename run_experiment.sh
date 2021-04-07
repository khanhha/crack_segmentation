#!/bin/bash
mkdir Experiment
mkdir Experiment/CRACK500
mkdir Experiment/CRACK500/test_results
mkdir Experiment/CRACK500/test_results/first
mkdir Experiment/CRACK500/test_results/second
mkdir Experiment/CRACK500/test_results/third
mkdir Experiment/CRACK500/model
mkdir Experiment/CRACK500/model/first
mkdir Experiment/CRACK500/model/second
mkdir Experiment/CRACK500/model/third

mkdir Experiment/CFD
mkdir Experiment/CFD/test_results
mkdir Experiment/CFD/test_results/first
mkdir Experiment/CFD/test_results/second
mkdir Experiment/CFD/test_results/third
mkdir Experiment/CFD/model
mkdir Experiment/CFD/model/first
mkdir Experiment/CFD/model/second
mkdir Experiment/CFD/model/third

# 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train Crack
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
{
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first
} 2>&1 | tee Experiment/Crack500/test_results/Crack0001First.log


python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.005
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.01
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.015
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.02
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.05
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.1
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.2

# Train CFD
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.001
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.005
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.01
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.015
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.02
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.05
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.1
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.2
}