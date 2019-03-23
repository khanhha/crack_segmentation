import numpy as np
import argparse
from pathlib import Path
import shutil
import os
import scipy.io as io
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os.path import join
from PIL import Image, ImageOps, ImageEnhance
from torchvision.transforms import RandomSizedCrop
from torchvision.transforms.functional import resized_crop
import tqdm

img_size = (448, 448)

def copy_forest(in_dir, out_dir, id):
    print(f'start copying {id}')

    forest_dir = os.path.join(in_dir, id)

    mask_dir = os.path.join(forest_dir,'groundTruth')
    img_dir = os.path.join(forest_dir,'image')

    dest_mask_dir = os.path.join(*[out_dir, 'masks'])
    dest_img_dir  = os.path.join(*[out_dir, 'images'])

    mask_names = set()
    for path in Path(mask_dir).glob('*.mat'):
        mat = io.loadmat(str(path))
        mask = mat['groundTruth']['Segmentation'][0,0]
        mask = (mask == 2).astype(np.uint8)*255
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        mask_names.add(path.stem)
        out_path = os.path.join(dest_mask_dir,f'{id}_{path.stem}.jpg')
        cv.imwrite(filename=out_path, img = mask)

    for path in Path(img_dir).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        out_path = os.path.join(dest_img_dir, f'{id}_{path.name}')
        img = cv.imread(str(path))
        img = cv.resize(img, dsize=img_size)
        cv.imwrite(filename=out_path, img = img)

    # for img_path in Path(dest_img_dir).glob('forest_*.jpg'):
    #     mask_path = os.path.join(dest_mask_dir, img_path.name)
    #     img = cv.imread(str(img_path))
    #     mask = cv.imread(str(mask_path))
    #     plt.subplot(131)
    #     plt.imshow(img)
    #     plt.subplot(132)
    #     plt.imshow(mask)
    #     plt.subplot(133)
    #     plt.imshow(img)
    #     plt.imshow(mask, alpha=0.3)
    #     plt.show()

def copy_cracktree200(in_dir, out_dir, id):
    print(f'start copying {id}')

    root_dir = os.path.join(in_dir, id)
    img_dir = os.path.join(root_dir,'cracktree200rgb')
    mask_dir = os.path.join(root_dir,'cracktree200_gt')

    mask_names = set()
    for path in Path(mask_dir).glob('*.png'):
        mask = cv.imread(str(path))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask >0).astype(np.uint8)*255
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        mask_names.add(path.stem)
        outpath = os.path.join(*[out_dir, 'masks', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = mask)

    for path in Path(img_dir).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        img = cv.imread(str(path))
        img = cv.resize(img, dsize=img_size)
        outpath = os.path.join(*[out_dir, 'images', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = img)

def copy_GAPS384(in_dir, out_dir, id):
    print(f'start copying {id}')

    root_dir = os.path.join(in_dir, id)
    img_dir = os.path.join(root_dir,'croppedimg')
    mask_dir = os.path.join(root_dir,'croppedgt')

    mask_names = set()
    for path in Path(mask_dir).glob('*.png'):
        mask = cv.imread(str(path))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask >0).astype(np.uint8)*255
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        #plt.imshow(mask)
        #plt.show()
        mask_names.add(path.stem)
        outpath = os.path.join(*[out_dir, 'masks', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = mask)

    for path in Path(img_dir).glob('*.jpg'):
        if path.stem not in mask_names:
            continue
        img = cv.imread(str(path))
        img = cv.resize(img, dsize=img_size)
        outpath = os.path.join(*[out_dir, 'images', f'{id}_{path.stem}.jpg'])
        cv.imwrite(filename=outpath, img = img)

def copy_CRACK500(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start copying {id}')

    root_dir = os.path.join(in_dir, id)

    subdir_names = ['testcrop', 'traincrop', 'valcrop']
    cnt = 0
    for subdir_name in subdir_names:
        dir = join(*[root_dir, subdir_name])

        img_paths = {}
        for path in Path(dir).glob('*.*'):
            if '_mask' in path.stem:
                continue
            if path.suffix == '.jpg' or path.suffix == '.JPG':
                img_paths[path.stem] = path

        for path in Path(dir).glob('*.png'):
            if path.stem not in img_paths:
                print(f'no image file for the mask {str(path)}')
                continue

            mask  = cv.imread(str(path))
            img_path = str(img_paths[path.stem])
            img = cv.imread(img_path)

            # import matplotlib.pyplot as plt
            # plt.subplot(131)
            # plt.imshow(img)
            # plt.subplot(132)
            # plt.imshow(mask)
            # plt.subplot(133)
            # plt.imshow(img)
            # plt.imshow(mask, alpha=0.5)
            # plt.title(path.stem)
            # plt.show()

            img = cv.resize(img, dsize=img_size)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255

            cv.imwrite(filename=join(*[out_dir_img,  f'{id}_{path.stem}.jpg']),  img = img)
            cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{path.stem}.jpg']), img = mask)

            cnt +=1

    print(f'copied {cnt} image-mask pairs from {id}')

def copy_CFD(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start copying {id}')

    img_dir = join(*[in_dir, id, 'cfd_image'])
    mask_dir = join(*[in_dir, id, 'cfd_gt', 'seg_gt'])

    mask_names = set(path.stem for path in Path(mask_dir).glob('*.png'))

    cnt= 0
    for path in Path(img_dir).glob('*.jpg'):

        if path.stem not in mask_names:
            print(f'no mask found for the image {path}')
            continue

        img  = cv.imread(str(path))
        mask = cv.imread(join(*[mask_dir, f'{path.stem}.png']))

        if mask.shape != img.shape:
            print(f'mismatched img-mask shape {name} : {img.shape} - {mask.shape}')
            continue

        img = cv.resize(img, dsize=img_size)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{path.stem}.jpg']), img=mask)

        cnt += 1

    print(f'copied {cnt} image-mask pairs from {id}')

def copy_DeepCrack(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start copying {id}')

    img_dir = join(*[in_dir, id, 'test_img'])
    mask_dir = join(*[in_dir, id, 'test_lab'])
    mask_names = set(path.stem for path in Path(mask_dir).glob('*.png'))

    cnt= 0
    for path in Path(img_dir).glob('*.jpg'):

        if path.stem not in mask_names:
            print(f'no mask found for the image {path}')
            continue

        img  = cv.imread(str(path))
        mask = cv.imread(join(*[mask_dir, f'{path.stem}.png']))

        if mask.shape != img.shape:
            print(f'mismatched img-mask shape {name} : {img.shape} - {mask.shape}')
            continue

        img = cv.resize(img, dsize=img_size)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{path.stem}.jpg']), img=mask)

        cnt += 1

    print(f'copied {cnt} image-mask pairs from {id}')

def copy_Sylvie_Chambon(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start copying {id}')

    img_dir = join(*[in_dir, id, 'img'])
    mask_dir = join(*[in_dir, id, 'masks_machine'])
    mask_names = set(path.stem for path in Path(mask_dir).glob('*.png'))

    cnt= 0
    for path in Path(img_dir).glob('*.jpg'):

        if path.stem not in mask_names:
            print(f'no mask found for the image {path}')
            continue

        img  = cv.imread(str(path))
        mask = cv.imread(join(*[mask_dir, f'{path.stem}.png']))

        if mask.shape != img.shape:
            print(f'mismatched img-mask shape {name} : {img.shape} - {mask.shape}')
            continue

        img = cv.resize(img, dsize=img_size)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)*255

        cv.imwrite(filename=join(*[out_dir_img,  f'{id}_{path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{path.stem}.jpg']), img=mask)

        cnt += 1

    print(f'copied {cnt} image-mask pairs from {id}')

def copy_noncrack(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start copying {id}')
    img_dir = join(*[in_dir, id, 'images'])
    mask_dir = join(*[in_dir, id, 'masks'])

    mask_names = set(path.stem for path in Path(mask_dir).glob('*.jpg'))

    cnt= 0
    for path in Path(img_dir).glob('*.jpg'):

        if path.stem not in mask_names:
            print(f'no mask found for the image {path}')
            continue

        img  = cv.imread(str(path))
        mask = cv.imread(join(*[mask_dir, f'{path.stem}.jpg']))

        if mask.shape != img.shape:
            print(f'mismatched img-mask shape {name} : {img.shape} - {mask.shape}')
            continue

        img = cv.resize(img, dsize=img_size, interpolation=cv.INTER_AREA)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{path.name}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{path.name}.jpg']), img=mask)

        cnt += 1

    print(f'copied {cnt} image-mask pairs from {id}')

def copy_Rissbilder_for_Florian(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start random cropping images under {id}')
    img_dir = os.path.join(*[in_dir, id, 'images'])
    mask_dir = os.path.join(*[in_dir, id, 'masks'])

    mask_paths = dict([(path.stem, path )for path in Path(mask_dir).glob('*.*')])

    img_paths = list([path for path in Path(img_dir).glob('*.*')])
    tq = tqdm.tqdm(total=len(img_paths))

    cnt = 0

    for img_path in Path(img_dir).glob('*.*'):

        if img_path.stem not in mask_paths:
            continue
        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_paths[img_path.stem]))

        img = cv.resize(img, dsize=img_size)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)*255

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{img_path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{img_path.stem}.jpg']), img=mask)

    print(f'copied {cnt} image-mask pairs from {id}')

    tq.close()

def copy_Volker(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start random cropping images under {id}')
    img_dir = os.path.join(*[in_dir, id, 'images'])
    mask_dir = os.path.join(*[in_dir, id, 'masks'])

    mask_paths = dict([(path.stem, path )for path in Path(mask_dir).glob('*.*')])

    img_paths = list([path for path in Path(img_dir).glob('*.*')])
    tq = tqdm.tqdm(total=len(img_paths))
    cnt = 0
    for img_path in Path(img_dir).glob('*.*'):
        if img_path.stem not in mask_paths:
            continue
        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_paths[img_path.stem]))

        img = cv.resize(img, dsize=img_size)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{img_path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{img_path.stem}.jpg']), img=mask)

    print(f'copied {cnt} image-mask pairs from {id}')

    tq.close()

def copy_Nohra(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start random cropping images under {id}')
    img_dir = os.path.join(*[in_dir, id, 'images'])
    mask_dir = os.path.join(*[in_dir, id, 'masks'])

    mask_paths = dict([(path.stem, path )for path in Path(mask_dir).glob('*.*')])

    img_paths = list([path for path in Path(img_dir).glob('*.*')])
    tq = tqdm.tqdm(total=len(img_paths))
    cnt = 0
    for img_path in Path(img_dir).glob('*.*'):
        if img_path.stem not in mask_paths:
            continue
        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_paths[img_path.stem]))

        img = cv.resize(img, dsize=img_size)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        cv.imwrite(filename=join(*[out_dir_img, f'{id}_{img_path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{img_path.stem}.jpg']), img=mask)

    print(f'copied {cnt} image-mask pairs from {id}')

    tq.close()

def copy_Eugen_Muller(in_dir, out_dir_img, out_dir_mask, id):
    print(f'start random cropping images under {id}')
    img_dir = os.path.join(*[in_dir, id, 'images'])
    mask_dir = os.path.join(*[in_dir, id, 'masks'])

    mask_paths = dict([(path.stem, path) for path in Path(mask_dir).glob('*.*')])

    img_paths = list([path for path in Path(img_dir).glob('*.*')])
    tq = tqdm.tqdm(total=len(img_paths))
    cnt=0
    for img_path in Path(img_dir).glob('*.*'):
        if img_path.stem not in mask_paths:
            continue
        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_paths[img_path.stem]))

        img = cv.resize(img, dsize=img_size)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        cv.imwrite(filename=join(*[out_dir_img,  f'{id}_{img_path.stem}.jpg']), img=img)
        cv.imwrite(filename=join(*[out_dir_mask, f'{id}_{img_path.stem}.jpg']), img=mask)

    tq.close()
    print(f'copied {cnt} image-mask pairs from {id}')


def rm_files(dir, pattern = '*.*'):
    for path in Path(dir).glob(pattern):
        os.remove(str(path))

def copy_files(src_dir, dst_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    rm_files(dst_dir)
    for name in names:
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        shutil.copy(src=src_path, dst=dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',type=str, help='input dataset directory')
    parser.add_argument('-out_dir', type=str, help='output dataset directory')
    args = parser.parse_args()

    img_dir = os.path.join(args.out_dir, 'images')
    mask_dir = os.path.join(args.out_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rm_files(img_dir)
    rm_files(mask_dir)

    id_6 = 'Rissbilder_for_Florian'
    copy_Rissbilder_for_Florian(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_6)

    id_11 = 'Sylvie_Chambon'
    copy_Sylvie_Chambon(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_11)

    id_10 = 'DeepCrack'
    copy_DeepCrack(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_10)

    id_7 = 'Volker'
    copy_Volker(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_7)
    id_8 = 'Eugen_Muller'
    copy_Eugen_Muller(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_8)
    id_9 = 'Nohra_Muller'
    copy_Nohra(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_9)
    id_10 = 'DeepCrack'
    copy_DeepCrack(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_10)
    id_0 = 'forest'
    copy_forest(args.in_dir, args.out_dir, id=id_0)
    id_1 = 'cracktree200'
    copy_cracktree200(args.in_dir, args.out_dir, id=id_1)
    id_2 = 'GAPS384'
    copy_GAPS384(args.in_dir, args.out_dir, id=id_2)
    id_3 = 'CRACK500'
    copy_CRACK500(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_3)
    id_4 = 'CFD'
    copy_CFD(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_4)
    id_5 = 'noncrack'
    copy_noncrack(args.in_dir, out_dir_img=img_dir, out_dir_mask=mask_dir, id=id_5)

    fnames = [path.name for path in Path(img_dir).glob('*.*')]
    labels = np.zeros(len(fnames), dtype=np.int)
    for i,name in enumerate(fnames):
        if id_0 in name:
            l = 1
        elif id_1 in name:
            l = 2
        elif id_2 in name:
            l = 3
        elif id_3 in name:
            l = 4
        elif id_4 in name:
            l = 5
        elif id_5 in name:
            l = 6
        elif id_6 in name:
            l = 7
        elif id_7 in name:
            l = 8
        else:
            l=0
        labels[i] = l

    train_names, test_names = train_test_split(fnames, test_size=0.15, stratify=labels)
    copy_files(img_dir, join(*[args.out_dir,  'train', 'images']), train_names)
    copy_files(mask_dir, join(*[args.out_dir, 'train', 'masks']), train_names)
    copy_files(img_dir, join(*[args.out_dir,  'test', 'images']), test_names)
    copy_files(mask_dir, join(*[args.out_dir, 'test', 'masks']), test_names)

