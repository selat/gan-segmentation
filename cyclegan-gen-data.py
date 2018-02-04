import skimage.io as skio
from skimage.transform import resize
import os
from tqdm import tqdm
import numpy as np

SIZE = 512

def central_crop(img):
    h, w, _ = img.shape
    startx = (w - h) // 2
    return img[:,startx:startx + h]

def gen_gta():
    path = '/mnt/data6T/data_ext/@GTA5_Datasets/Playing_for_Data/images/{:05}.png'
    mask_path = '/mnt/data6T/data_ext/@GTA5_Datasets/Playing_for_Data/labels/{:05}.png'
    out_path = 'datasets/gta2cityscapes/trainA/{:05}.png'
    out_mask_path = 'datasets/gta2cityscapes/maskA/{:05}.png'

    for i in tqdm(range(1, 1001)):
        img = skio.imread(path.format(i))
        mask = skio.imread(mask_path.format(i))
        img = resize(central_crop(img), (SIZE, SIZE))
        mask = resize(central_crop(mask), (SIZE, SIZE), preserve_range=True, order=0).astype(np.uint8)
        skio.imsave(out_path.format(i), img)
        skio.imsave(out_mask_path.format(i), mask)

def gen_cityscapes():
    path = '/mnt/data1T2/datasets2/Cityscapes_Dataset/data/gtFine_trainvaltest.unzip/gtFine/cityscapes-gtfine-person/'
    out_path = 'datasets/gta2cityscapes/trainB/{:05}.png'
    files = filter(lambda x : x[-8:] != '-msk.png', os.listdir(path))
    for i in tqdm(range(1000)):
        fname = files[i]
        img = skio.imread(os.path.join(path, fname))
        img = resize(central_crop(img), (SIZE, SIZE))
        skio.imsave(out_path.format(i), img)

gen_gta()
gen_cityscapes()