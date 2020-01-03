#!/bin/bash

python inference.py \
-model_path ./models/model_weights_cracknausnet.pt \
-image_path ./uav75/test_img/DSC00595.jpg \
-out_path ./results/DSC00595.jpg \
-device cuda:0
