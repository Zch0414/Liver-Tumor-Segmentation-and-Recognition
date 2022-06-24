import os
from os.path import splitext
from glob import glob
import scipy.ndimage as ndimage
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from .dataset import BasicDataset as FatherDataset

def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 3)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image


class BasicDataset(Dataset):
    def __init__(self, filepath_dic, trans=False):
        self.img_dir = filepath_dic['image']
        self.lab_dir = filepath_dic['label']
        self.trans = trans
        self.img_ids = FatherDataset.dir2ids(self.img_dir)
        self.lab_ids = FatherDataset.dir2ids(self.lab_dir)

        # The code below is only used to test the whole code.
        # self.img_ids = self.img_ids[0:4]
        # self.lab_ids = self.lab_ids[0:4]

        assert len(self.img_ids) == len(self.lab_ids), \
            "The size of img_ids and lab_ids should be same"

        self.len_ = len(self.img_ids)

    def __len__(self):
        return self.len_

    @classmethod
    def preprocess(cls, img_nd):
        assert len(img_nd.shape) == 3, \
            "The size of the img_nd is wrong"
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    @classmethod
    def transform(cls, image):
        if random.random() > 0.5:
            image = random_rot_flip(image)
        elif random.random() > 0.5:
            image = random_rotate(image)

        image = cls.preprocess(image)
        return image

    def __getitem__(self, i):
        img_i = self.img_ids[i]
        lab_i = self.lab_ids[i]
        id_img = os.path.split(img_i)[-1].split("_")[1:]
        id_lab = os.path.split(lab_i)[-1].split("_")[1:]
        assert id_img == id_lab, \
            "The image id {0} and the label id {1} are not corresponding".\
                format(id_img, id_lab)

        img = glob(self.img_dir + img_i + '.npy')
        lab = glob(self.lab_dir + lab_i + '.npy')

        assert len(img) == 1, \
            "Either no mask or multiple masks {2} found for the ID {0}_{1}". \
                format(id_img[0], id_img[1], len(img))

        assert len(lab) == 1, \
            "Either no mask or multiple masks {2} found for the ID {0}_{1}". \
                format(id_lab[0], id_lab[1], len(lab))

        image = np.load(img[0])
        label = np.load(lab[0])

        if self.trans:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        sample = {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor),
            'name': img_i
        }
        return sample

