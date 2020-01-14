#!/bin/bash

python inference.py \
--model_path ./models/model_weights_cracknausnet.pt \
--image_path ./uav75/test_img/DSC00568b.jpg \
--out_path ./results/DSC00568b.jpg \
--device cuda:0
#--planking_filtering \
#--rotation_fusion \

