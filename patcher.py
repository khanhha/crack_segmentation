import numpy as np

def split_in_chunks(img, size=448, padding=0):
  h,w,c = img.shape
  print("shape: ", h, w)

  steps_h = int(h/size) + 1
  steps_w = int(w/size) + 1
  print("then size: ", steps_h, steps_w)

  img_tmp = np.zeros((steps_h*size, steps_w*size,c))
  img_tmp[0:h,0:w,:] = img.copy()

  patches = []

  for i in range(steps_h):
    for j in range(steps_w):
      patch = img_tmp[i*size:(i+1)*size, j*size:(j+1)*size,:]
      patch = np.pad(patch, ((padding,padding),(padding,padding),(0, 0)), 'constant')
      patches.append(np.uint8(patch))

  return patches


def merge_from_chunks(patches, targ_h, targ_w, size=448, padding=0):
  steps_h = int(targ_h/size) + 1
  steps_w = int(targ_w/size) + 1

  img_merged = np.zeros((steps_h*size, steps_w*size))

  for i in range(steps_h):
    for j in range(steps_w):
      patch = patches[i*steps_w+j]
      patch = patch[padding:-padding,padding:-padding]
      img_merged[i*size:(i+1)*size, j*size:(j+1)*size] = patch

  img_merged = img_merged[0:targ_h,0:targ_w]

  return img_merged
