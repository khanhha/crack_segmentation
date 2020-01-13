from scipy.ndimage.filters import maximum_filter
import numpy as np


def fuse_results(img_crack1, img_crack2):
  # apply average fusion
  return img_crack1*0.5 + img_crack2*0.5


def filter_cracks(img_crack, img_plank, thresh=0.5):
  # max filtering
  kernel = np.ones((10,10))
  img_tmp = maximum_filter(img_plank, footprint=kernel)
 
  # create binary mask
  img_bin = np.where(img_tmp>=thresh,0,1)

  # apply mask on crack image
  img_res = img_bin*img_crack
  
  return img_res*255
