import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import argparse
from scipy import signal


image_path="./results/DSC07158_original.jpg"
crack_path="./results/DSC07158_crack.jpg"
plank_path="./results/DSC07158_planking.jpg"


def fuse_results(img_crack1, img_crack2):
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


def filter_probs(img_crack, thresh=0.5, kernel_size=10):
  img_crack = img_crack/255

  s = kernel_size
#  kernel = np.ones((10,10)) / (10*10)
#  img_tmp = signal.convolve2d(img_crack, kernel, boundary='symm', mode='same') 
  if False:
    kernel = np.ones((s,s),np.float32)/ (s*s)
    img_tmp = cv2.filter2D(img_crack,-1,kernel)
  else:
    img_tmp = cv2.blur(img_crack,(s,s))



  img_bin = np.where(img_tmp>=thresh,1,0)

  plt.subplot(221)
  plt.imshow(img_crack)
  plt.subplot(222)
  plt.imshow(img_tmp)
  plt.subplot(223)
  plt.imshow(img_bin)
#  plt.show()

  img_res = img_bin*img_crack

#  if out_path is not None:
#    cv2.imwrite(out_path, img_res*255)
  
  return img_bin



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--crack_path', type=str, required=True, help='crack image to be filtered')
  parser.add_argument('--plank_path', type=str, required=True, help='planking image to aid the filtering')
  parser.add_argument('--out_path', type=str, required=True, help='resulting image')
  args = parser.parse_args()

  # load image
  img_crack = cv2.imread(args.crack_path, cv2.IMREAD_GRAYSCALE)/255.0
  img_plank = cv2.imread(args.plank_path, cv2.IMREAD_GRAYSCALE)/255.0

  filter_cracks(img_crack, img_plank, args.out_path)



if False:
  # load image
  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  img_crack = cv2.imread(crack_path, cv2.IMREAD_GRAYSCALE)/255.0
  img_plank = cv2.imread(plank_path, cv2.IMREAD_GRAYSCALE)/255.0

  kernel = np.ones((10,10))#np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
  img_tmp = maximum_filter(img_plank, footprint=kernel)

  img_blur = gaussian_filter(img_tmp, sigma=7)

  img_bin = np.where(img_tmp>=0.5,0,1)

  f = np.fft.fft2(img_crack)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum1 = 20*np.log(np.abs(fshift))

  f = np.fft.fft2(img_plank)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum2 = 20*np.log(np.abs(fshift))

  f = np.fft.fft2(img_tmp)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum3 = 20*np.log(np.abs(fshift))

  f = np.fft.fft2(img_blur)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum4 = 20*np.log(np.abs(fshift))

  plt.subplot(2,7,1)
  plt.imshow(img)
  plt.subplot(2,7,2)
  plt.imshow(img_crack)
  plt.subplot(2,7,3)
  plt.imshow(img_plank)
  plt.subplot(2,7,4)
  plt.imshow(img_tmp)
  plt.subplot(2,7,5)
  plt.imshow(img_blur)
  plt.subplot(2,7,6)
  plt.imshow(img_bin)
  plt.subplot(2,7,7)
  plt.imshow((1-img_tmp)*img_crack)
  plt.subplot(2,7,8)
  plt.imshow((1-img_blur)*img_crack)
  plt.subplot(2,7,9)
  plt.imshow(img_bin*img_crack)
  plt.subplot(2,7,10)
  plt.imshow(magnitude_spectrum1, cmap = 'gray')
  plt.subplot(2,7,11)
  plt.imshow(magnitude_spectrum2, cmap = 'gray')
  plt.subplot(2,7,12)
  plt.imshow(magnitude_spectrum3, cmap = 'gray')
  plt.subplot(2,7,13)
  plt.imshow(magnitude_spectrum4, cmap = 'gray')
  #plt.show()


  plt.subplot(3,2,1)
  plt.imshow(img_tmp)
  plt.subplot(3,2,2)
  plt.imshow(img_blur)
  plt.subplot(3,2,3)
  plt.imshow(img_crack)
  plt.subplot(3,2,4)
  plt.imshow((1-img_tmp)*img_crack)
  plt.subplot(3,2,5)
  plt.imshow((1-img_blur)*img_crack)
  plt.subplot(3,2,6)
  plt.imshow(img_bin*img_crack)
  plt.show()
