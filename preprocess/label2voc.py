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
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'), exist_ok=True)
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

    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        # if '9S6A2822' not in label_file:
        #      continue

        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.output_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                args.output_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                args.output_dir, 'SegmentationClassPNG', base + '.png')
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
            ##

            labelme.utils.lblsave(out_png_file, lbl)

            plt.clf()
            plt.imshow(img)
            plt.imshow(lbl, alpha=0.4)
            plt.savefig(out_viz_file)
            #plt.show()

            # viz = labelme.utils.draw_label(
            #     lbl, img, class_names, colormap=colormap)
            # PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()