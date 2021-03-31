
mkdir CRACK500/test
mkdir CRACK500/test/images
mkdir CRACK500/test/masks
mkdir CRACK500/train
mkdir CRACK500/train/images
mkdir CRACK500/train/masks

mkdir cfd/test
mkdir cfd/test/images
mkdir cfd/test/masks

mkdir cfd/train
mkdir cfd/train/images
mkdir cfd/train/masks

mv crack_segmentation_dataset/train/images/CRACK500* CRACK500/train/images
mv crack_segmentation_dataset/test/images/CRACK500* CRACK500/test/images
mv crack_segmentation_dataset/train/masks/CRACK500* CRACK500/train/masks &&
mv crack_segmentation_dataset/test/masks/CRACK500* CRACK500/test/masks

mv crack_segmentation_dataset/train/images/CFD* cfd/train/images &&
mv crack_segmentation_dataset/test/images/CFD* cfd/test/images
mv crack_segmentation_dataset/train/masks/CFD* cfd/train/masks &&
mv crack_segmentation_dataset/test/masks/CFD* cfd/test/masks

conda create --name crack
conda activate crack
conda install -c anaconda pytorch-gpu && \
conda install -c conda-forge opencv && \
conda install matplotlib scipy numpy tqdm pillow && \
conda install numba

pip install torchvision
pip install torchgeometry