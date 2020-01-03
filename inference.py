"""
Usage: python inference.py [model_path] [image_path] [out_path] [device]

Run inference of the cracknausnet model on one specified image
"""

import numpy as np
import argparse
import torch
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from TernausNet.unet_models import UNet16

class CrackDetector():
  def __init__(self, model_path='', image_path='', out_path='./results/', device='cuda:0'):
    self.module_name = 'crack detector'
    self.model_path = model_path
    self.image_path = image_path
    self.out_path = out_path
    self.device = torch.device(device)


  def run(self):
    # load model
    model = UNet16(pretrained=True, num_classes=3, is_deconv=True)
    checkpoint = torch.load(self.model_path)
    model.load_state_dict(checkpoint['model'])

    # move to device (GPU) and set mode to evaluation
    model.to(self.device)
#    model.cuda()
    model.eval()

    # load image
    img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
    h_orig, w_orig, _ = img.shape

    # make divisible by 32 to use fully convolutional property
    h_tmp = 32*round(h_orig / 32)
    w_tmp = 32*round(w_orig / 32)
    img_tmp = cv2.resize(img, (w_tmp, h_tmp), cv2.INTER_AREA)
    img_tmp = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2RGB)
   

    # apply preprocessing transformations (values from ImageNet)
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    tfms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(channel_means, channel_stds)
      ])
    X = tfms(img_tmp)

    # add batch dimension
    X = Variable(X.unsqueeze(0)).to(self.device)  # [N, C, H, W]

    # compute forward pass
    mask = model(X)
   
    # apply softmax on output
    mask = F.softmax(mask, dim=1).data.cpu().numpy()

    # select relevant class (mask[0,0,:,:] non-crack, mask[0,1,:,:] is planking pattern)
    mask = mask[0,2,:,:]

    # shape back to original size
    mask = cv2.resize(mask, (w_orig, h_orig), cv2.INTER_AREA)

    # save heatmap
    cv2.imwrite(self.out_path, mask*255)

    # save overlayed image (225/2 for nicer use of color map)
    mask = np.uint8(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255/2)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    cv2.imwrite(self.out_path.replace('.jpg', '_overlay.jpg'), cv2.addWeighted(img, 0.9, mask, 0.8, 0))
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_path', type=str, default='./models/model_weights_cracknausnet.pt', required=False, help='path to model weights')
  parser.add_argument('-image_path', type=str, default='./uav75/test_img/DSC00595.jpg', required=False, help='path to input image')
  parser.add_argument('-out_path', type=str, default='./results/DSC00595.jpg', required=False, help='path to which the prediction is saved')
  parser.add_argument('-device', type=str, default='cuda:0', required=False, help='device to used for inference')
  args = parser.parse_args()

  detector = CrackDetector(args.model_path, args.image_path, args.out_path, args.device)
  detector.run()


