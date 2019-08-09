import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
from skimage.transform import hough_line
from scipy.stats import entropy
import scipy.cluster.hierarchy as hcluster

# load image
img_orig = cv2.imread('/home/chrisbe/images/cracks/benchmark/images_cropped/DSC00844.jpg')#594.jpg')
img = cv2.imread('/home/chrisbe/repos/crack_segmentation/bench_khanh_2_out/DSC00844.jpg')
img_copy = img.copy()


# compute gradient image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,130,180,apertureSize = 3)


# compute hough transform
hspace, angles, distances = hough_line(edges, theta=np.linspace(-np.pi / 16, np.pi / 16, 3))#22.5))

# get proper data format (array of indices)
size = hspace.shape[0]
tmp = np.linspace(0,size-1,size, int)[np.sum(hspace, axis=1) > 100]
data = np.array([tmp, tmp]).T


# clustering without predefined number of cluster
thresh = 50.5
try: 
  clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
except:
  clusters = []

print(clusters)


# loop over the found clusters
for c in np.unique(clusters):
  print("Current cluster: ", c)

  # get current cluster
  tmp = clusters[clusters==c]

  # consider only clusters with more than 3 elements
  if len(tmp) < 3:
    continue
  
  # get the angle (and entropy: how certain is the angle)
  theta = angles[np.argmax(np.sum(hspace[data[clusters==c,0].astype(int),:], axis=0))]
  theta_entrop = entropy(np.sum(hspace[data[clusters==c,0].astype(int),:], axis=0))

  # get the distance (mean and range)
  rho = np.mean(distances[data[clusters==c,0].astype(int)])
  width = np.max(distances[data[clusters==c,0].astype(int)]) - np.min(distances[data[clusters==c,0].astype(int)])
 
  # some console output
  print("\ttheta_entrop: ", theta_entrop)
  print("\ttheta: ", theta)
  print("\trho: ", rho)
  print("\twidth: ", width)

  # skip if too uncertain about the angle
  if theta_entrop > 0.4:
    continue

  # construct and draw line
  a = math.cos(theta)
  b = math.sin(theta)
  x0 = a * rho
  y0 = b * rho
  pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
  pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))
  cv2.line(img, pt1, pt2, (10,10,10), 4*int(width), cv2.LINE_AA)


# display output
plt.subplot(221)
plt.imshow(img_orig)
plt.subplot(222)
plt.imshow(img)
plt.subplot(223)
plt.imshow(img_copy)
plt.subplot(224)
plt.imshow(img)

plt.show()

