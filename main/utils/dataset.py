from torch.utils.data import Dataset
import random
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
import torch
import os
from os import listdir
from os.path import splitext
from glob import glob
from PIL import Image

class BasicDataset(Dataset):
    def __init__(
            self,
            file_path_dic,
            grid=7,
            trans=True,
            scale=1
    ):
        """
        :param file_path_dic:
        The path for data file,
        which should be a dictionary including data and label
        example = {
            'image': '../data_train/image/',
            'seg_lab': '../data_train/label/',
            'cls_lab': '../cls_lab_train/'
        }
        :param grid:
        Whether to use patch classification or not. Default: True
        :param trans:
        Whether to use data transform or not.
        True for training dataset and False for testing dataset. Default: True
        :param scale:
        The scale parameter that will be used to scale the input data.
        Default: 1.
        """
        self.img_dir = file_path_dic['image']
        self.seg_lab_dir = file_path_dic['seg_lab']
        self.cls_lab_dir = file_path_dic['cls_lab']
        self.grid = grid
        self.scale = scale
        self.trans = trans
        self.img_ids = self.dir2ids(self.img_dir)
        self.seg_lab_ids = self.dir2ids(self.seg_lab_dir)
        self.cls_lab_ids = self.dir2ids(self.cls_lab_dir)

        # The code below is only used to test the whole code.

        # self.img_ids = self.img_ids[0: 1000: 4]
        # self.seg_lab_ids = self.seg_lab_ids[0: 1000: 4]
        # self.cls_lab_ids = self.cls_lab_ids[0: 1000: 4]

        assert 0 < scale <= 1, "Scale, must be between 0 and 1."
        assert len(self.img_ids) == len(self.seg_lab_ids) == len(self.cls_lab_ids), \
            "The size of img_ids and seg_lab_ids and cls_lab_ids should be same."

        self._len = len(self.img_ids)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        img_i = self.img_ids[i]
        seg_lab_i = self.seg_lab_ids[i]
        cls_lab_i = self.cls_lab_ids[i]
        id_img = os.path.split(img_i)[-1].split("_")[1:]
        id_seg_lab = os.path.split(seg_lab_i)[-1].split("_")[1:]
        id_cls_lab = os.path.split(cls_lab_i)[-1].split("_")[1:]
        assert id_img == id_seg_lab == id_cls_lab, \
            "The image id is {0}, the segment label id is {1}, the classification label is is {2}".\
                format(id_img, id_seg_lab, id_cls_lab)

        path_img = glob(self.img_dir + img_i + '.png')
        path_seg_lab = glob(self.seg_lab_dir + seg_lab_i + '.png')
        path_cls_lab = glob(self.cls_lab_dir + cls_lab_i + '.png')

        assert len(path_img) == 1, \
            "Either no mask or multiple masks {2} found for the ID {0}_{1}". \
                format(id_img[0], id_img[1], len(path_img))

        assert len(path_seg_lab) == 1, \
            "Either no segment label or multiple label {2} found for the ID {0}_{1}". \
                format(id_seg_lab[0], id_seg_lab[1], len(path_seg_lab))

        assert len(path_cls_lab) == 1, \
            "Either no classification label or multiple label {2} found for the ID {0}_{1}". \
                format(id_cls_lab[0], id_cls_lab[1], len(path_cls_lab))

        img = Image.open(path_img[0])
        seg_lab = Image.open(path_seg_lab[0])
        cls_lab = Image.open(path_cls_lab[0])

        if self.scale != 1:
            img = zoom(img, (self.scale, self.scale), order=3)
            seg_lab = zoom(seg_lab, (self.scale, self.scale, 1), order=1)

        if self.trans:
            img, seg_lab, cls_lab = self.transforms(img, seg_lab, cls_lab, self.grid)
            img = self.preprocess(img)
            seg_lab = self.preprocess(seg_lab)
            cls_lab = self.preprocess(cls_lab, cls_grid=self.grid)
            sample = {
                'image': img,
                'seg_label': seg_lab,
                'cls_label': cls_lab,
                'name': img_i
            }

        else:
            img = self.preprocess(img)
            seg_lab = self.preprocess(seg_lab)
            cls_lab = self.preprocess(cls_lab, cls_grid=self.grid)
            sample = {
                'image': img,
                'seg_label': seg_lab,
                'cls_label': cls_lab,
                'name': img_i
            }
        return sample

    @classmethod
    def random_rot_flip(cls, image, seg_label, cls_label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        seg_label = np.rot90(seg_label, k)
        cls_label = np.rot90(cls_label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        seg_label = np.flip(seg_label, axis=axis).copy()
        cls_label = np.flip(cls_label, axis=axis).copy()
        return image, seg_label, cls_label

    @classmethod
    def random_rotate(cls, image, seg_label, cls_label):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        seg_label = ndimage.rotate(seg_label, angle, order=0, reshape=False)
        cls_label = ndimage.rotate(cls_label, angle, order=0, reshape=False)
        return image, seg_label, cls_label

    @classmethod
    def dir2ids(cls, root):
        ids = []
        for dir in listdir(root):
            _dir = os.path.join(root, dir)
            ids.extend(
                [os.path.join(dir, splitext(_f)[0]) for _f in listdir(_dir)]
            )
            ids.sort()
            # ids : */label_*_*
        return ids

    @classmethod
    def preprocess(cls, data_pil, cls_grid=7):
        data_nd = np.array(data_pil)
        if len(data_nd.shape) == 2: # which means it is the input image
            data_nd = (data_nd - 99.40) / 39.96
            data_nd = np.expand_dims(data_nd, axis=0)
        elif len(data_nd.shape) == 3: # which means it is the seg_label or cls_label
            data_nd = data_nd[:, :, (2, 1, 0)]
            if cls_grid == 1: # which means it is the cls_label
                new_cls_lab = np.zeros(3)
                if np.any(data_nd, axis=(0, 1))[2]:
                    new_cls_lab[2] = 1
                elif np.any(data_nd, axis=(0, 1))[1]:
                    new_cls_lab[1] = 1
                else:
                    new_cls_lab[0] = 1
                return torch.from_numpy(new_cls_lab).type(torch.FloatTensor)
            data_nd = data_nd.transpose((2, 0, 1)) / 255
        return torch.from_numpy(data_nd).type(torch.FloatTensor)

    @classmethod
    def transforms(cls, image, seg_label, cls_label, cls_grid=7):
        if random.random() > 0.5:
            if cls_grid != 1:
                image, seg_label, cls_label = cls.random_rot_flip(image, seg_label, cls_label)
            else:
                image, seg_label, _ = cls.random_rot_flip(image, seg_label, seg_label)
        elif random.random() > 0.5:
            if cls_grid != 1:
                image, seg_label, cls_label = cls.random_rotate(image, seg_label, cls_label)
            else:
                image, seg_label, _ = cls.random_rotate(image, seg_label, seg_label)

        return image, seg_label, cls_label

# if __name__ == '__main__':
#     file_pth_dic = {
#         'image': '../data_train/image/',
#         'seg_lab': '../data_train/label/',
#         'cls_lab': '../class_lab_train/'
#     }
#
#     dataset = BasicDataset(file_pth_dic)
#     print(dataset)





















