import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_loss import DiceLoss, calculate_dice
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import numpy as np
import copy
import argparse


def cls_patch_output(pred):
  pred_softmax = F.softmax(pred, dim=1)
  pred_id = torch.argmax(pred_softmax, dim=1).squeeze(0).cpu().detach().numpy().reshape(-1)
  # pred_score and pred_id are on cpu
  pred_score = pred_softmax[:, 2, ...].cpu().detach().numpy().reshape(-1)
  # pred_id (49,) pred_score(49,)
  return list(pred_score), list(pred_id.astype(int))

def cls_slice_output(pred):
  pred_softmax = F.softmax(pred, dim=1)
  pred_id = torch.argmax(pred_softmax, dim=1).squeeze(0).cpu()
  # slice_id and pred_score are on cpu
  slice_id = pred_id.max().detach().cpu()
  pred_softmax = pred_softmax.squeeze(0).cpu().detach()
  pred_score = pred_softmax[2, pred_id == 2].detach()
  if len(pred_score) == 0:
    pred_score = torch.tensor(0).float()
  else:
    pred_score = pred_score.float().mean()

  return list([pred_score.numpy()]), list([slice_id.numpy().astype(int)])

def seg_slice_eval(pred, label):
  pred_id = torch.argmax(F.softmax(pred, dim=1), dim=1).squeeze(0)
  pred_id = pred_id.cpu().detach().numpy()
  consistency_seg_pred = pred_id.max()
  # pred and label are on cpu
  dice_tumor = calculate_dice(pred_id == 2, label[2, :, :])
  dice_liver = calculate_dice(pred_id == 1, label[1, :, :])
  out = F.one_hot(torch.from_numpy(pred_id), num_classes=3).permute(2, 0, 1)
  return dice_tumor, dice_liver, out, consistency_seg_pred

def find_fpr_tpr(fprs, tprs, thresholds):
  index = np.argmin(np.absolute(thresholds - 0.5))
  return fprs[index], tprs[index]

def cls_eval(true_patch, pred_patch, score_patch, true_slice, pred_slice, score_slice):
  true_patch = np.array(true_patch)
  pred_patch = np.array(pred_patch)
  true_slice = np.array(true_slice)
  pred_slice = np.array(pred_slice)
  acc_patch = accuracy_score(pred_patch, true_patch)
  acc_slice = accuracy_score(pred_slice, true_slice)
  true_patch[true_patch < 2] = 0
  true_slice[true_slice < 2] = 0
  fprs_patch, tprs_patch, thresholds_patch = roc_curve(true_patch, score_patch, pos_label=2)
  fprs_slice, tprs_slice, thresholds_slice = roc_curve(true_slice, score_slice, pos_label=2)
  fpr_patch, tpr_patch = find_fpr_tpr(fprs_patch, tprs_patch, thresholds_patch)
  fpr_slice, tpr_slice = find_fpr_tpr(fprs_slice, tprs_slice, thresholds_slice)
  auc_patch = auc(fprs_patch, tprs_patch)
  auc_slice = auc(fprs_slice, tprs_slice)
  patch_result = {
    "acc": acc_patch, # perform as a 3-classes
    "fpr": fpr_patch, # perform as a binary-classes
    "tpr": tpr_patch,
    "auc": auc_patch
  }
  slice_result = {
    "acc": acc_slice,
    "fpr": fpr_slice,
    "tpr": tpr_slice,
    "auc": auc_slice
  }
  return patch_result, slice_result

def eval_single_volume(net, net_type, loader, device, writer, epoch):

  net.eval()
  n_val = len(loader)

  consistency_score = 0
  tumor_dice = 0
  liver_dice = 0
  pred_patch = []
  pred_slice = []
  score_patch = []
  score_slice = []
  true_patch = []
  true_slice = []



  with tqdm(total=n_val, desc='Validation round', unit='img', leave=True) as pbar:
    for batch in loader:

      img = batch['image']
      seg_lab = batch['seg_label']
      cls_lab = batch['cls_label']
      file_name = batch['name']
      img = img.to(device=device, dtype=torch.float32)
      # seg_lab and cls_lab are on cpu
      seg_lab = seg_lab.squeeze(0).cpu().detach().numpy()  
      cls_lab = cls_lab.argmax(dim=1).squeeze(0).cpu().detach().numpy().reshape(-1)   

      if net_type == 'join':
        with torch.no_grad():
          cls_pred, seg_pred = net(img)
          cls_slice_lab = cls_lab.max()
          patch_score, patch_pred = cls_patch_output(cls_pred)
          slice_score, slice_pred = cls_slice_output(cls_pred)
          pred_patch.extend(patch_pred)
          score_patch.extend(patch_score)
          pred_slice.extend(slice_pred)
          score_slice.extend(slice_score)
          true_patch.extend(list(cls_lab.astype(int)))
          true_slice.extend(list([cls_slice_lab.astype(int)]))
          dice_tumor, dice_liver, out, consistency_seg_pred = seg_slice_eval(seg_pred, seg_lab)
          tumor_dice += dice_tumor
          liver_dice += dice_liver
          consistency_score += slice_pred[0] == consistency_seg_pred
          mask = torch.cat((torch.from_numpy(seg_lab), out.cpu().detach()), dim=1)
          mask[0, ...] = 0
          if writer is not None:
            writer.add_image('Masks/test_epoch{0}/{1}_{2}_{3}_{4}_{5}'.format(epoch, file_name, dice_liver, dice_tumor, consistency_seg_pred, slice_pred[0]), mask, epoch)

      if net_type == 'class':
        with torch.no_grad():
          cls_slice_lab = cls_lab.max()
          cls_pred = net(img)
          patch_score, patch_pred = cls_patch_output(cls_pred)
          slice_score, slice_pred = cls_slice_output(cls_pred)
          pred_patch.extend(patch_pred)
          score_patch.extend(patch_score)
          pred_slice.extend(slice_pred)
          score_slice.extend(slice_score)
          true_patch.extend(list(cls_lab.astype(int)))
          true_slice.extend(list([cls_slice_lab.astype(int)]))

      if net_type == 'segment':
        with torch.no_grad():
          seg_pred = net(img)
          dice_tumor, dice_liver, out, _ = seg_slice_eval(seg_pred, seg_lab)
          tumor_dice += dice_tumor
          liver_dice += dice_liver
          mask = torch.cat((torch.from_numpy(seg_lab), out.cpu().detach()), dim=1)
          mask[0, :, :] = 0
          if writer is not None:
            writer.add_image('Masks/test_epoch{0}/{1}_{2}_{3}'.format(epoch, file_name, liver_dice, tumor_dice), mask, epoch)

      
      pbar.update(1)

    if net_type == 'join':
      cls_result_patch, cls_result_slice = cls_eval(true_patch, pred_patch, score_patch, true_slice, pred_slice, score_slice)
      tumor_dice = tumor_dice / n_val
      liver_dice = liver_dice / n_val
      consistency_score = consistency_score / n_val
      if writer is not None:
        writer.add_scalar('Patch/Accuracy/test', cls_result_patch['acc'], epoch)
        writer.add_scalar('Patch/FPR/test', cls_result_patch['fpr'], epoch)
        writer.add_scalar('Patch/TPR/test', cls_result_patch['tpr'], epoch)
        writer.add_scalar('Patch/AUC/test', cls_result_patch['auc'], epoch)
        writer.add_scalar('Slice/Accuracy/test', cls_result_slice['acc'], epoch)
        writer.add_scalar('Slice/FPR/test', cls_result_slice['fpr'], epoch)
        writer.add_scalar('Slice/TPR/test', cls_result_slice['tpr'], epoch)
        writer.add_scalar('Slice/AUC/test', cls_result_slice['auc'], epoch)
        writer.add_scalar('Dice/Tumor', tumor_dice, epoch)
        writer.add_scalar('Dice/Liver', liver_dice, epoch)
      return cls_result_patch, cls_result_slice, tumor_dice, liver_dice, consistency_score

    if net_type == 'class':
      cls_result_patch, cls_result_slice = cls_eval(true_patch, pred_patch, score_patch, true_slice, pred_slice, score_slice)
      if writer is not None:
        writer.add_scalar('Patch/Accuracy/test', cls_result_patch['acc'], epoch)
        writer.add_scalar('Patch/FPR/test', cls_result_patch['fpr'], epoch)
        writer.add_scalar('Patch/TPR/test', cls_result_patch['tpr'], epoch)
        writer.add_scalar('Patch/AUC/test', cls_result_patch['auc'], epoch)
        writer.add_scalar('Slice/Accuracy/test', cls_result_slice['acc'], epoch)
        writer.add_scalar('Slice/FPR/test', cls_result_slice['fpr'], epoch)
        writer.add_scalar('Slice/TPR/test', cls_result_slice['tpr'], epoch)
        writer.add_scalar('Slice/AUC/test', cls_result_slice['auc'], epoch)
      return cls_result_patch, cls_result_slice

    if net_type == 'segment':
      tumor_dice = tumor_dice / n_val
      liver_dice = liver_dice / n_val
      if writer is not None:
        writer.add_scalar('Dice/Tumor', tumor_dice, epoch)
        writer.add_scalar('Dice/Liver', liver_dice, epoch)
      return tumor_dice, liver_dice

def get_test_args():

  parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--net_type', type=str, default='join', dest='net_type')  # could be modified with class and segmentation
  parser.add_argument('--scale', type=float, default=1, dest='img_scale')
  parser.add_argument('--grid', type=float, default=3, dest='grid')  # grid can only equal to 1, 3, 5, 7, re-run cls_lab_eng.py before you want to change the grid
  parser.add_argument('--residual', type=bool, default=False, dest='residual')
  parser.add_argument('--load', type=str, default=False, dest='load')  # could be modified with the path of .path file
  return parser.parse_args()

if __name__ == '__main__':
  from utils import BasicDataset
  from torch.utils.data import DataLoader
  from models import swinYNet
  import logging

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  args = get_test_args()
  if args.net_type != 'join':
    assert args.residual is not True, "residual is designed only for join!"
  args.load = './result/model/join_grid_3.path'
  val_path_dic = {
    'image': '../data_test/image/',
    'seg_lab': '../data_test/label/',
    'cls_lab': '../class_lab_test/'
  }
  val_set = BasicDataset(
    file_path_dic=val_path_dic,
    grid=args.grid,
    trans=False,
    scale=1
  )
  val_loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True
  )

  cls_flag, seg_flag = False, False
  if args.net_type == 'class':
    cls_flag = True
    seg_flag = False
  elif args.net_type == 'segment':
    cls_flag = False
    seg_flag = True
  else:
    cls_flag = True
    seg_flag = True
  net = swinYNet(cls=cls_flag, seg=seg_flag, in_channels=1, num_classes=3, grid=args.grid, residual=args.residual)
  net.load_state_dict(torch.load(args.load, map_location=device))
  net.to(device=device)
  if args.net_type == 'join':
    cls_result_patch, cls_result_slice, tumor_dice, liver_dice, consistency_score = eval_single_volume(net, args.net_type, val_loader, device, None, 0)
    logging.info(
      '''
      Live dice:         {0}
      Tumor dice:        {1}
      Patch accuracy:    {2}
      Slice accuracy:    {3}
      consistency_score: {4}
      '''.format(liver_dice, tumor_dice, cls_result_patch['acc'], cls_result_slice['acc'], consistency_score)
    )
  if args.net_type == 'class':
    cls_result_patch, cls_result_slice = eval_single_volume(net, args.net_type, val_loader, device, None, 0)
    logging.info(
      '''
      Patch accuracy: {0}
      Slice accuracy: {1}
      '''.format(cls_result_patch['acc'], cls_result_slice['acc'])
    )
  if args.net_type == 'segment':
    tumor_dice, liver_dice = eval_single_volume(net, args.net_type, val_loader, device, None, 0)
    logging.info(
      '''
      Live dice:      {0}
      Tumor dice:     {1}
      '''.format(liver_dice, tumor_dice)
    )