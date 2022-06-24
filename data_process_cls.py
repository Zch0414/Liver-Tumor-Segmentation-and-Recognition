import os
import sys
sys.path.append("..")
import torch
from unet.unet_model import UNet
import cv2
from torch.utils.data import DataLoader
from utils.dataset import BasicDataset
from tqdm import tqdm
from os.path import splitext
import numpy as np
import SimpleITK as sitk
from os import listdir
from PIL import Image
from glob import glob
from scipy.ndimage import zoom

data_root = './data'
train_path_dic = {
    'image': './data_train/image/',
    'label': './data_train/label/'
}
val_path_dic = {
    'image': './data_test/image/',
    'label': './data_test/label/'
}

model_path = './checkpoints/model_result/unet_lambda_0.1_dice_90.98.pth'

save_img_path = './data_cls/image/'
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
save_lab_path = './data_cls/label/'
if not os.path.exists(save_lab_path):
    os.mkdir(save_lab_path)


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def image_generator(
        train_path_dic,
        val_path_dic,
        save_img_path,
        net,
        need_train=True,
        dice=100.0,
        img_scale=0.5,
        prob=True,
        cat=True
):

    save_img_path = os.path.join(save_img_path, 'dice_{0}'.format(dice))
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    save_train = os.path.join(save_img_path, 'train')
    if not os.path.exists(save_train):
        os.mkdir(save_train)
    save_test = os.path.join(save_img_path, 'test')
    if not os.path.exists(save_test):
        os.mkdir(save_test)

    dataset_train = BasicDataset(train_path_dic, img_scale, trans=False)
    dataset_val = BasicDataset(val_path_dic, img_scale, trans=False)
    if need_train:
        n = len(dataset_train) + len(dataset_val)
    else:
        n = len(dataset_val)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    net.eval()
    with tqdm(total=n, desc='Image generator', unit='image') as pbar:
        if need_train:
            for batch in train_loader:
                dir = os.path.split(batch['name'][0])[0]
                dir = os.path.join(save_train, dir)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                file_name = os.path.split(batch['name'][0])[1]
                file_path = os.path.join(dir, file_name)
                img = batch['image'].to(device=device, dtype=torch.float32)
                label = batch['label'].to(device=device, dtype=torch.float32)
                if dice == 100 or net is None:
                    if cat is True:
                        pred = label
                    else:
                        pred = img * label
                else:
                    with torch.no_grad():
                        pred = net(img)
                    pred = torch.sigmoid(pred)
                    if prob is False:
                        pred = pred > 0.5
                    if cat is True:
                        pred = pred * 255
                    else:
                        pred = pred.to(device=device, dtype=torch.float32)
                        pred = pred * img
                output = pred.squeeze(0).cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite('%s.png' % file_path, output, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                pbar.update(1)

        for batch in val_loader:
            dir = os.path.split(batch['name'][0])[0]
            dir = os.path.join(save_test, dir)
            if not os.path.exists(dir):
                os.mkdir(dir)
            file_name = os.path.split(batch['name'][0])[1]
            file_path = os.path.join(dir, file_name)
            img = batch['image'].to(device=device, dtype=torch.float32)
            label = batch['label'].to(device=device, dtype=torch.float32)
            if dice == 100 or net is None:
                if cat is True:
                    pred = label
                else:
                    pred = img * label
            else:
                with torch.no_grad():
                    pred = net(img)
                pred = torch.sigmoid(pred)
                if prob is False:
                    pred = pred > 0.5
                if cat is True:
                    pred = pred * 255
                else:
                    pred = pred.to(device=device, dtype=torch.float32)
                    pred = pred * img
            output = pred.squeeze(0).cpu().numpy().transpose((1, 2, 0))
            cv2.imwrite('%s.png' % file_path, output, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            pbar.update(1)

def label_generator(data_root, save_lab_path):
    _len = len(listdir(data_root))

    with tqdm(total=_len, desc='Data process', unit='file') as pbar:
        for _f in listdir(data_root):

            _id = int(_f.split(".")[0])
            if _id > 27 and _id < 48:
                path_save = os.path.join(save_lab_path, 'test')
            else:
                path_save = os.path.join(save_lab_path, 'train')

            if not os.path.exists(path_save):
                os.mkdir(path_save)
            path_save = os.path.join(path_save, str(_id))

            path = os.path.join(data_root, _f)
            path_seg = os.path.join(path, listdir(path)[0])
            seg = sitk.ReadImage(path_seg, sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            z = np.any(seg_array, axis=(1, 2))
            _start, _end = np.where(z)[0][[0, -1]]
            _start = max(0, _start - 5)
            _end = min(seg_array.shape[0] - 1, _end + 5)

            seg_array = seg_array[_start: _end + 1, :, :]
            label_list = []

            for i in range(0, _end - _start + 1):
                seg_img = seg_array[i, :, :]
                if np.max(seg_img) >= 2:
                    label_list.append(1)
                else:
                    label_list.append(0)
            label_array = np.array(label_list)
            np.save(path_save, label_array)

            pbar.update(1)

def image_process_cat(dice, _str, _size, GOP, prob='False'):
    root_mask = './data_cls/image/dice_{0}_prob_{1}_cat_True/{2}/'.format(dice, prob, _str)
    root_img = './data_{0}/image/'.format(_str)
    save_path = './data_cls/{0}/'.format(_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_root = os.path.join(save_path, 'image')
    if not os.path.exists(save_path_root):
        os.mkdir(save_path_root)
    for _d in listdir(root_mask):
        mask_dir = os.path.join(root_mask, _d)
        img_dir = os.path.join(root_img, _d)
        id = 'image_{}_'.format(_d)

        n_mask = len(listdir(mask_dir))
        n_img = len(listdir(img_dir))
        assert n_mask == n_img, \
            "The mask directory and the image directory are not matched"
        n = n_mask

        save_path = os.path.join(save_path_root, _d)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with tqdm(total=n, desc='The {0}th file'.format(_d), unit='img', leave=True) as pbar:
            for index in range(n):
                if index > n-GOP:
                    break
                save_name = 'volume_{0}_{1}'.format(_d, index)
                merge = np.zeros((_size, _size, 2 * GOP)) # HWC
                for i in range(GOP):
                    file_name = id + str(index+i)  # 'image_0_0'
                    file_path_mask = os.path.join(mask_dir, file_name + '.png')
                    mask = Image.open(file_path_mask)
                    file_path_img = os.path.join(img_dir, file_name + '.png')
                    img = Image.open(file_path_img)
                    if img.size != (_size, _size):
                        img = zoom(img, (_size / img.size[0], _size / img.size[1]), order=3)
                    img_nd = np.array(img)
                    mask_nd = np.array(mask)
                    merge[:, :, i] = img_nd
                    merge[:, :, i + GOP] = mask_nd

                save_dir = os.path.join(save_path, save_name)
                np.save(save_dir, merge)
                pbar.update(1)

def image_process(dice, _str, _size, GOP, prob='False'):
    root_img = './data_cls/image/dice_{0}_prob_{1}_cat_False/{2}/'.format(dice, prob, _str)
    save_path = './data_cls/{0}/'.format(_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_root = os.path.join(save_path, 'image')
    if not os.path.exists(save_path_root):
        os.mkdir(save_path_root)
    for _d in listdir(root_img):
        mask_dir = os.path.join(root_img, _d)
        id = 'image_{}_'.format(_d)

        n = len(listdir(mask_dir))

        save_path = os.path.join(save_path_root, _d)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with tqdm(total=n, desc='The {0}th file'.format(_d), unit='img', leave=True) as pbar:
            for index in range(n):
                if index > n-GOP:
                    break
                save_name = 'volume_{0}_{1}'.format(_d, index)
                merge = np.zeros((_size, _size, GOP)) # HWC
                for i in range(GOP):
                    file_name = id + str(index+i)  # 'image_0_0'
                    file_path_img = os.path.join(mask_dir, file_name + '.png')
                    img = Image.open(file_path_img)
                    img_nd = np.array(img)
                    merge[:, :, i] = img_nd

                save_dir = os.path.join(save_path, save_name)
                np.save(save_dir, merge)
                pbar.update(1)


def label_process(_str, GOP):
    root = './data_cls/label/{0}/'.format(_str)
    save_path = './data_cls/{0}/'.format(_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_root = os.path.join(save_path, 'label')
    if not os.path.exists(save_path_root):
        os.mkdir(save_path_root)
    for _d in listdir(root):
        save_path = os.path.join(save_path_root, splitext(_d)[0])
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        label_array = np.load(os.path.join(root, _d))
        label_list = list(label_array)
        for index in range(len(label_list)):
            if index > len(label_list) - GOP:
                break
            save_name = 'label_{0}_{1}'.format(splitext(_d)[0], index)
            label = np.array([0])
            for i in range(GOP):
                if label_list[index + i] > 0:
                    label = np.array([1])
                    break
            save_dir = os.path.join(save_path, save_name)
            np.save(save_dir, label)



if __name__ == '__main__':
    # dice = 100.0
    dice = splitext(os.path.split(model_path)[-1])[0].split("_")[-1]
    print(dice)
    GOP = 1

    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)

    # image_generator(
    #     train_path_dic,
    #     val_path_dic,
    #     save_img_path,
    #     net,
    #     need_train=True,
    #     dice=dice,
    #     prob=False,
    #     cat=False
    # )

    # label_generator(data_root, save_lab_path)

    image_process(dice, 'train', 256, GOP)
    image_process(dice, 'test', 256, GOP)

    # image_process_cat(dice, 'train', 256, GOP)
    # image_process_cat(dice, 'test', 256, GOP)
    #
    label_process('test', GOP)
    label_process('train', GOP)