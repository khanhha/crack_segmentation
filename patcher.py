import numpy as np

def split_in_chunks(img, size=448, pad=0):
  h,w,c = img.shape

  # number of required patches
  steps_h = int(h/size) + 1
  steps_w = int(w/size) + 1

  # create 
  img_tmp = np.zeros((steps_h*size + 2*pad, steps_w*size + 2*pad,c))
  img_tmp[pad:h+pad,pad:w+pad,:] = img.copy()

  # container for patches
  patches = []

  # loop over all patches
  for i in range(steps_h):
    for j in range(steps_w):
      # obtain patch with padding and append to results
      patch = img_tmp[i*size:(i+1)*size+2*pad, j*size:(j+1)*size+2*pad,:]
      patches.append(np.uint8(patch))

  return patches


def merge_from_chunks(patches, targ_h, targ_w, size=448, pad=0):
  # determine temporary size
  steps_h = int(targ_h/size) + 1
  steps_w = int(targ_w/size) + 1

  # create dummy for result
  img_merged = np.zeros((steps_h*size, steps_w*size))

  # loop over all patches
  for i in range(steps_h):
    for j in range(steps_w):
      # get patch and remove padding
      patch = patches[i*steps_w+j]
      patch = patch[pad:-pad,pad:-pad]
     
      # paste patch into results
      img_merged[i*size:(i+1)*size, j*size:(j+1)*size] = patch

  # remove irrelevant regions
  img_merged = img_merged[0:targ_h,0:targ_w]

  return img_merged
