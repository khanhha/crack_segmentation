# Crack Segmentation

For cloning the repository, please use the recursive flag to also clone the submodules:
```
git clone https://github.com/khanhha/crack_segmentation.git --recursive
```

## Model
The trained model can be obtained by:
```
cd models
wget https://cloud.uni-weimar.de/s/m5KpTHomWjnyGQb/download -O model_weights_cracknausnet.pt
```

## Inference
In order to run inference on one image, please use the provided bash script
```
bash run_inference.sh
```
or apply, where each argument should be self-exanatory:
```
python inference.py \
-model_path ./models/model_weights_cracknausnet.pt \
-image_path ./uav75/test_img/DSC00595.jpg \
-out_path ./results/DSC00595.jpg \
-device cuda:0
```

## Drawbacks
- Overproportionally large encoder (VGG16)
- Generalizability to other concrete surfaces (with possibly different planking patterns) not assessed
- Still sensitive to planking patterns
- Applicable only to smaller crack (i.e. cracks lesser than 8?->check pixels wide)
- Presumably limited applicability of the TernausNet for problems with 3+ classes



## Reference
For reference and citation please use (will be shortly put into final format):

Benz, C., Debus, P., Ha, H.-K. & Rodehorst, V. (2019): Crack Segmentation on UAS-based Imagery using Transfer Learning. Image and Vision Computing New Zealand, IVCNZ, Dunedin.
