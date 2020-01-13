#!/bin/bash

python inference.py \
--model_path ./models/model_weights_cracknausnet.pt \
--image_path /home/chrisbe/tmp/9_Kamera5_23_DxO.jpg \
--out_path ./results/9_Kamera5_23_DxO.jpg \
--planking_filtering \
--rotation_fusion \
--device cuda:0 

