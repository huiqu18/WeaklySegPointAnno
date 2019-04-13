"""
We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load (image, label_voronoi, label_cluster) items from given directory lists.

"""

import torch.utils.data as data
import os
from PIL import Image
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img


# get the image list pairs
def get_imgs_list(dir_list, post_fix=None):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label_vor.png', 'label_cluster.png',...]
    :return: e.g. [(img1.png, img1_label_vor.png, img1_label_cluster.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list
    if len(dir_list) != len(post_fix) + 1:
        raise (RuntimeError('Should specify the postfix of each img type except the first input.'))

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img),]
        for i in range(1, len(img_filename_list)):
            img_name = '{:s}_{:s}'.format(img1_name, post_fix[i-1])
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


# dataset that supports multiple images
class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None, loader=img_loader):
        """
        :param dir_list: [img_dir, label_voronoi_dir, label_cluster_dir]
        :param post_fix:  ['label_vor.png', 'label_cluster.png']
        :param num_channels:  [3, 3, 3]
        :param data_transform: data transformations
        :param loader: image loader
        """
        super(DataFolder, self).__init__()
        if len(dir_list) != len(post_fix) + 1:
            raise (RuntimeError('Length of dir_list is different from length of post_fix + 1.'))
        if len(dir_list) != len(num_channels):
            raise (RuntimeError('Length of dir_list is different from length of num_channels.'))

        self.img_list = get_imgs_list(dir_list, post_fix)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

