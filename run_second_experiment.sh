#!/bin/bash
# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train Crack Second Run (Log files are named as follows: Crack<decimals LR>Second.log
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack001Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack005Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack01Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack015Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack02Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack05Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack1Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/second -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/second
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/second > Experiment/CRACK500/test_results/Crack2Second.log
sudo rm -r ./Experiment/CRACK500/model/second/*
sudo rm -r ./Experiment/CRACK500/test_results/second/*

# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train CFD Second Run (Log files are named as follows: CFD<decimals LR>Second.log
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD001Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD005Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD01Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD015Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD02Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD05Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD1Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/second -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/second/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/second
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/second > Experiment/CFD/test_results/CFD2Second.log
sudo rm -r ./Experiment/CFD/model/second/*
sudo rm -r ./Experiment/CFD/test_results/second/*