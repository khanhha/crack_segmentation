#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys
import  matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import argparse
import base64
import json
import os
import labelme
from labelme import utils
import cv2 as cv
from PIL import Image, ImageOps, ImageEnhance
from torchvision.transforms import RandomSizedCrop
from torchvision.transforms.functional import resized_crop, crop, resize
from tqdm import tqdm

def random_crop(img, mask, size, scale=(0.2, 0.8), ratio=(3. / 4., 4. / 3.), n_tries = 10, crack_px_percent = 0.3, resize=False):
    n_total_crack = np.sum((mask > 0)[:])

    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    results = []

    img_w = img.size[0]
    img_h = img.size[1]

    for i in range(n_tries):
        i, j, h, w = RandomSizedCrop.get_params(img, scale, ratio)
        sub_img  = resized_crop(img, i, j, h, w, size, Image.BILINEAR)
        sub_mask = resized_crop(mask,i, j, h, w, size, Image.NEAREST)
        sub_img = np.asarray(sub_img)
        sub_mask = np.asarray(sub_mask)

        tmp = np.asarray(img.crop((j, i, j + w, i + h)))
        n_crack_pixels = np.sum((tmp>0)[:])

        crk_ratio = float(n_crack_pixels)/n_total_crack
        if crk_ratio < crack_px_percent:
            print('missing')
            continue

        results.append((sub_img, sub_mask, (i, j, h, w)))

    _img  = np.asarray(resized_crop(img, 0, 0, img_h, img_w, size, Image.BILINEAR))
    _mask = np.asarray(resized_crop(mask, 0, 0, img_h, img_w, size, Image.NEAREST))
    _img  = np.asarray(_img)
    _mask = np.asarray(_mask)

    results.append((_img, _mask, (0, 0, img_h, img_w)))

    return results

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    # if osp.exists(args.output_dir):
    #     print('Output directory already exists:', args.output_dir)
    #     sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'masks'), exist_ok=True)
    #os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'), exist_ok=True)
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = labelme.utils.label_colormap(255)

    for label_file in tqdm(list([path for path in glob.glob(osp.join(args.input_dir, '*.json'))])):
        #if '9S6A2822' not in label_file:
        #     continue

        #print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.output_dir, 'images', base + '.jpg')
            out_lbl_file = osp.join(
                args.output_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                args.output_dir, 'masks', base + '.jpg')
            out_viz_file = osp.join(
                args.output_dir,
                'SegmentationClassVisualization',
                base + '.jpg',
            )

            data = json.load(f)

            ##
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(label_file), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {'_background_': 0}
            for shape in sorted(data['shapes'], key=lambda x: x['label']):
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            #lb = cv.imread(join(*[args.label_dir, f'{path.stem}.png']))
            #lb = cv.cvtColor(lb, cv.COLOR_BGR2GRAY)
            lbl = (lbl > 0).astype(np.uint8) * 255
            lbl = cv.morphologyEx(src=lbl, op=cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_RECT, (20, 20)))

            results = random_crop(img, lbl, size=(448, 448), n_tries=10)
            #tq.update(1)
            for sub_img, sub_mask, crop_info in results:
                info = f'{crop_info[0]}_{crop_info[1]}_{crop_info[2]}_{crop_info[3]}'
                cv.imwrite(filename=os.path.join(*[args.output_dir, 'images', f'{base}_{info}.jpg']), img=sub_img)
                cv.imwrite(filename=os.path.join(*[args.output_dir, 'masks',  f'{base}_{info}.jpg']), img=sub_mask)
                #cnt += 1
                plt.clf()
                plt.imshow(sub_img)
                plt.imshow(sub_mask, alpha=0.4)
                plt.savefig(osp.join(args.output_dir, 'SegmentationClassVisualization', f'{base}_{info}' + '.jpg',))

            #labelme.utils.lblsave(out_png_file, lbl)
            #np.save(out_lbl_file, lbl)
            #PIL.Image.fromarray(img).save(out_img_file)


            #plt.show()

            # viz = labelme.utils.draw_label(
            #     lbl, img, class_names, colormap=colormap)
            # PIL.Image.fromarray(viz).save(out_viz_file)

if __name__ == '__main__':
    main()