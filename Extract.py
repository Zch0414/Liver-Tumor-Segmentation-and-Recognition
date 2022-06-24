import sys
sys.path.append("..")
import os
from os import listdir
from os.path import splitext
import zipfile
from tqdm import tqdm

PATH_ZIP = '../LITS17/'
PATH_TGT_ROOT = './data'
if not os.path.exists(PATH_TGT_ROOT):
    os.mkdir(PATH_TGT_ROOT)


def extract(path_zip, path_tgt_root):
    with tqdm(total=len(listdir(PATH_ZIP)), desc='Extracting', unit='file') as pbar:
        for parents, dirnames, filenames in os.walk(path_zip):
            for _f in filenames:
                if splitext(_f)[1] != '.zip':
                    continue
                name = splitext(_f)[0]
                _id = name.split("-")[-1]
                path_tgt = os.path.join(path_tgt_root, _id)
                if not os.path.exists(path_tgt):
                    os.mkdir(path_tgt)
                path = os.path.join(parents, _f) #../LITS17/segmentation-*.nii.zip
                _file_zip = zipfile.ZipFile(path)
                file = _file_zip.namelist()[0] #segmentation-0.nii
                if file not in listdir(path_tgt):
                    _file_zip.extractall(path=path_tgt)
                _file_zip.close()
                pbar.update(1)
    print('Done!')

extract(PATH_ZIP, PATH_TGT_ROOT)