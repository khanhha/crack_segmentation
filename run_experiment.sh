#!/bin/bash
sudo rm -r ./Experiment

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

# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train Crack First Run (Log files are named as follows: Crack<decimals LR>First.log
python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack001First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack005First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack01First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack015First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack02First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack05First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack1First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

python train_unet.py -data_dir ./CRACK500/train -model_dir ./Experiment/CRACK500/model/first -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CRACK500/test/images -model_path ./Experiment/CRACK500/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CRACK500/test_results/first
python evaluate_unet.py -ground_truth_dir ./CRACK500/test/masks -pred_dir ./Experiment/CRACK500/test_results/first > Experiment/CRACK500/test_results/Crack2First.log
sudo rm -r ./Experiment/CRACK500/model/first/*

# LR Rates: 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2

# Train CFD First Run (Log files are named as follows: CFD<decimals LR>First.log
python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.001
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD001First.log
sudo rm -r ./Experiment/CFD/model/first/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.005
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD005First.log
sudo rm -r ./Experiment/CFD/model/first/*

python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.01
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD01First.log
sudo rm -r ./Experiment/CFD/model/first/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.015
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD015First.log
sudo rm -r ./Experiment/CFD/model/first/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.02
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD02First.log
sudo rm -r ./Experiment/CFD/model/first/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.05
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD05First.log
sudo rm -r ./Experiment/CFD/model/first/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.1
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD1First.log
sudo rm -r ./Experiment/CFD/model/first/*


python train_unet.py -data_dir ./CFD/train -model_dir ./Experiment/CFD/model/first -model_type resnet34 -lr 0.2
python inference_unet.py -img_dir ./CFD/test/images -model_path ./Experiment/CFD/model/first/model_best.pt -model_type resnet34 -out_pred_dir ./Experiment/CFD/test_results/first
python evaluate_unet.py -ground_truth_dir ./CFD/test/masks -pred_dir ./Experiment/CFD/test_results/first > Experiment/CFD/test_results/CFD2First.log
sudo rm -r ./Experiment/CFD/model/first/*