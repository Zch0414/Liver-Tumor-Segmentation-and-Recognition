from torch.utils.data import Dataset
import random
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
import torch
from os import listdir
import os
from os.path import splitext
from glob import glob
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class BasicDataset(Dataset):
    def __init__(self, filepath_dic, scale=1, trans=False):
        self.img_dir = filepath_dic['image']
        self.lab_dir = filepath_dic['label']
        self.scale = scale
        self.trans = trans
        self.img_ids = self.dir2ids(self.img_dir)
        self.lab_ids = self.dir2ids(self.lab_dir)

        # The code below is only used to test the whole code.
        # self.img_ids = self.img_ids[0:4]
        # self.lab_ids = self.lab_ids[0:4]

        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        assert len(self.img_ids) == len(self.lab_ids), \
            "The size of img_ids and lab_ids should be same"

        self.len_ = len(self.img_ids)
    @classmethod
    def dir2ids(cls, root):
        ids = []
        for dir in listdir(root):
            _dir = os.path.join(root, dir)
            ids.extend(
                [os.path.join(dir, splitext(_f)[0]) for _f in listdir(_dir)]
            )
            ids.sort()
        return ids

    def __len__(self):
        return self.len_

    @classmethod
    def preprocess(cls, pil_img, name):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) ==2:
            img_nd = np.expand_dims(img_nd, axis=2)
        img_trans = img_nd.transpose((2, 0 ,1))
        if name == 'label':
            img_trans = img_trans / 255
        return img_trans

    @classmethod
    def transform(cls, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = cls.preprocess(image, 'image')
        label = cls.preprocess(label, 'label')

        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        sample['image'], sample['label'] = image, label
        return sample

    def __getitem__(self, i):
        img_i = self.img_ids[i]
        lab_i = self.lab_ids[i]

        id_img = os.path.split(img_i)[-1].split("_")[1:]
        id_lab = os.path.split(lab_i)[-1].split("_")[1:]
        assert id_img == id_lab, \
            "The image id {0} and the label id {1} are not corresponding".\
                format(id_img, id_lab)

        img = glob(self.img_dir + img_i + '.png')
        lab = glob(self.lab_dir + lab_i + '.png')


        assert len(img) == 1, \
            "Either no mask or multiple masks {2} found for the ID {0}_{1}".\
                format(id_img[0], id_img[1], len(img))

        assert len(lab) == 1, \
            "Either no mask or multiple masks {2} found for the ID {0}_{1}".\
                format(id_lab[0], id_lab[1], len(lab))

        image = Image.open(img[0])
        label = Image.open(lab[0])
        if self.scale != 1:
            image = zoom(image, (self.scale, self.scale), order=3)  
            label = zoom(label, (self.scale, self.scale), order=0)

        if self.trans:
            sample = {
                'image': image,
                'label': label,
                'name': img_i
            }
            sample = self.transform(sample)
        else:
            image = self.preprocess(image,'image')
            label = self.preprocess(label, 'label')
            sample = {
                'image': torch.from_numpy(image).type(torch.FloatTensor),
                'label': torch.from_numpy(label).type(torch.FloatTensor),
                'name': img_i
            }
        return sample


# if __name__ == '__main__':
#     filepath_dic = {
#         'image' : '../data_train/image/',
#         'label' : '../data_train/label/'
#     }
#     DataSet = BasicDataset(filepath_dic)
#     print(DataSet)
