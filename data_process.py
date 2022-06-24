import os
from os import listdir
from os.path import splitext
import sys
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import scipy.ndimage as ndimage
import cv2

PATH_TRAIN = './data_train/'
if not os.path.exists(PATH_TRAIN):
    os.mkdir(PATH_TRAIN)
PATH_TEST = './data_test/'
if not os.path.exists(PATH_TEST):
    os.mkdir(PATH_TEST)
ROOT = './data'

def data_process(path_root, path_train, path_test):

    path_train_label = os.path.join(path_train, 'label')
    if not os.path.exists(path_train_label):
        os.mkdir(path_train_label)
    path_train_img = os.path.join(path_train, 'image')
    if not os.path.exists(path_train_img):
        os.mkdir(path_train_img)
    path_test_label = os.path.join(path_test, 'label')
    if not os.path.exists(path_test_label):
        os.mkdir(path_test_label)
    path_test_img = os.path.join(path_test, 'image')
    if not os.path.exists(path_test_img):
        os.mkdir(path_test_img)

    _len = len(listdir(path_root))

    with tqdm(total=_len, desc='Data process', unit='file') as pbar:
        for _f in listdir(path_root):

            _id = int(_f.split(".")[0])
            if _id > 27 and _id < 48:
                path_save = path_test
            else:
                path_save = path_train

            save_seg = os.path.join(path_save, 'label', str(_id))
            if not os.path.exists(save_seg):
                os.mkdir(save_seg)
            save_ct = os.path.join(path_save, 'image', str(_id))
            if not os.path.exists(save_ct):
                os.mkdir(save_ct)
            path = os.path.join(path_root, _f)
            path_seg = os.path.join(path, listdir(path)[0])
            path_vol = os.path.join(path, listdir(path)[1])
            ct = sitk.ReadImage(path_vol, sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            seg = sitk.ReadImage(path_seg, sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            seg_array[seg_array > 0] = 255
            ct_array[ct_array > 250] = 250
            ct_array[ct_array < -200] = -200

            # if you want more data, use the code below. But it might cause over-fitting issue
            # ct_array = ndimage.zoom(
            #     ct_array,
            #     (ct.GetSpacing()[-1] / 1, 1, 1),
            #     order=3
            # )
            # seg_array = ndimage.zoom(
            #     seg_array,
            #     (ct.GetSpacing()[-1] / 1, 1, 1),
            #     order=0
            # )

            z = np.any(seg_array, axis=(1, 2))
            _start, _end = np.where(z)[0][[0, -1]]
            _start = max(0, _start - 5)
            _end = min(seg_array.shape[0] - 1, _end + 5)

            ct_array = ct_array[_start : _end + 1, :, :]
            seg_array = seg_array[_start : _end + 1, :, :]

            for i in range(0, _end - _start + 1):
                ct_img = ct_array[i, :, :]
                seg_img = seg_array[i, :, :]
                name_ct = "image_{0}_{1}".format(_id, i)
                name_seg = "label_{0}_{1}".format(_id, i)
                path_save_ct = os.path.join(save_ct, name_ct + '.png')
                path_save_seg = os.path.join(save_seg, name_seg + '.png')
                cv2.imwrite(path_save_ct, ct_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                cv2.imwrite(path_save_seg, seg_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])



            pbar.update(1)

if __name__ == '__main__':
    data_process(ROOT, PATH_TRAIN, PATH_TEST)
