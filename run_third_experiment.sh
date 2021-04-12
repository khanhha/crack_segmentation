#!/bin/bash
# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train Crack Third Run (Log files are named as follows: Crack<decimals LR>Third.log
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack001Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack005Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack01Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack015Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack02Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack05Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack1Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/third -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/third
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/third > Experiment/CRACK500/test_results/Crack2Third.log
sudo rm -r ./Experiment/CRACK500/model/third/*
sudo rm -r ./Experiment/CRACK500/test_results/third/*

# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train CFD Third Run (Log files are named as follows: CFD<decimals LR>Third.log
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD001Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD005Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD01Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD015Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD02Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD05Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD1Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/third -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/third/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/third
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/third > Experiment/CFD/test_results/CFD2Third.log
sudo rm -r ./Experiment/CFD/model/third/*
sudo rm -r ./Experiment/CFD/test_results/third/*