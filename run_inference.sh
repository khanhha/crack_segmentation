#!/bin/bash

python inference.py \
--model_path ./models/model_weights_cracknausnet.pt \
--image_path /home/chrisbe/tmp/DSC07158.jpg \
--out_path ./results/DSC07158.jpg \
--device cuda:0 \
--category 2
