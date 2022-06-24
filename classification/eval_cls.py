from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import torch
from tqdm import tqdm
import numpy as np
import os
import cv2

def find_fpr_tpr(fprs, tprs, thresholds):
    index = np.argmin(np.absolute(thresholds - 0.5))
    return fprs[index], tprs[index]

def eval_net(net, loader, device):
    prediction = []
    true = []
    score = []
    net.eval()

    n = len(loader)
    with tqdm(total=n, desc='Validation round', unit='img', leave=True) as pbar:
        for batch in loader:
            img = batch['image'].to(device=device, dtype=torch.float32)
            lab = batch['label'].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = net(img)
            pred = torch.sigmoid(pred).cpu().numpy()
            pred = pred.reshape(pred.shape[0])

            score.extend(list(pred))
            prediction.extend(list((pred > 0.5).astype(int)))
            true.extend(list(lab.cpu().numpy().astype(int)))
            pbar.update(img.shape[0])

        fprs, tprs, thresholds = roc_curve(np.array(true), np.array(score), pos_label=1)
        fpr ,tpr = find_fpr_tpr(fprs, tprs, thresholds)
        acc = accuracy_score(prediction, true)
        _auc = auc(fprs, tprs)

        net.train()

        return fpr, tpr, _auc, acc

def compare_net(net_1, net_2, loader_1, loader_2, device):
    """
    the function output the name list that net_1 can predict correct while the net_2 cannot
    Parameters
    ----------
    net_1: this should be your better model
    net_2: this should be your baseline model
    loader_1: this should be the data loader which is compatible with net_1
    loader_2: this should be the data loader which is compatible with net_2
    device

    Returns: the name list that net_1 can predict correct while the net_2 cannot
    -------

    """

    result_path = './compare_result.txt'
    TP = ['TP']
    TN = ['TN']

    net_1.eval()
    net_2.eval()
    assert len(loader_1) == len(loader_2),\
        "Then size of two loaders should be same"
    n = len(loader_1)

    with tqdm(total=n, desc='Validation round', unit='img', leave=True) as pbar:
        for batch_1, batch_2 in zip(loader_1, loader_2):
            img_1 = batch_1['image'].to(device=device, dtype=torch.float32)
            lab_1 = batch_1['label'].to(device=device, dtype=torch.float32)
            img_name_1 = batch_1['name']
            img_2 = batch_2['image'].to(device=device, dtype=torch.float32)
            lab_2 = batch_2['label'].to(device=device, dtype=torch.float32)
            img_name_2 = batch_2['name']

            assert img_1.shape[0] == 1, \
                "The batch size of the validation dataset should be 1"
            assert img_name_1 == img_name_2, \
                "Image's name in two loaders should be same"
            assert lab_1 == lab_2,\
                "The label in two loaders should be same"

            img_name = img_name_1
            lab = lab_1

            with torch.no_grad():
                pred_1 = net_1(img_1)
                pred_2 = net_2(img_2)
            pred_1 = torch.sigmoid(pred_1) > 0.5
            pred_2 = torch.sigmoid(pred_2) > 0.5
            pred_1 = pred_1.cpu().numpy()[0]
            pred_2 = pred_2.cpu().numpy()[0]
            lab = lab.cpu().numpy().astype(int)[0]

            if lab == 1 and pred_1 == lab and pred_2 != lab:
                TP.append(img_name)
            if lab == 0 and pred_1 == lab and pred_2 != lab:
                TN.append(img_name)
            pbar.update(img_1.shape[0])

        str = '\n'
        f = open(result_path, "w")
        f.write(str.join(TP))
        f.write(str.join(TN))
        f.close()

        return TP, TN