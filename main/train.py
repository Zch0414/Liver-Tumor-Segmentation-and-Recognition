import argparse
import logging
import os
import sys
import cv2
import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import swinYNet

from eval import eval_single_volume
import random



train_path_dic = {
  'image': '../data_train/image/',
  'seg_lab': '../data_train/label/',
  'cls_lab': '../class_lab_train/'
}

val_path_dic = {
  'image': '../data_test/image/',
  'seg_lab': '../data_test/label/',
  'cls_lab': '../class_lab_test/'
}

checkpoint_dir = './checkpoints/'
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)



def train_net(
  save_ep,
  net,
  net_type,
  device,
  epochs=5,
  batch_size=1,
  lr=0.001,
  dice_lam=0,
  seg_lam=0,
  grid=True,
  img_scale=1,
  save_cp=True,
  seed=0
):
  """
  net_type should be "join" or "class" or "segement"
  """
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = True
  from utils import DiceLoss, BasicDataset

  train_set = BasicDataset(
    file_path_dic = train_path_dic,
    grid=grid,
    trans=True,
    scale=img_scale
  )
  val_set = BasicDataset(
    file_path_dic = val_path_dic,
    grid=grid,
    trans=False,
    scale=img_scale
  )

  n_train, n_val = len(train_set), len(val_set)
  
  train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
  )
  val_loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True
  )

  writer = SummaryWriter(comment='LR_{0}_BS_{1}_DICELAM_{2}_SEGLAM{3}'.format(lr, batch_size, dice_lam, seg_lam))

  logging.info(
    '''
    Starting training: 
    Epochs:                        {0}
    Batch size:                    {1}
    Learning rate:                 {2}
    Training size:                 {3}
    Validation size:               {4}
    Lambda value for dice loss:    {5}
    Lambda value for segment loss: {6}
    Checkpoints:                   {7}
    Device:                        {8}
    Images scaling:                {9}
    Grid:                          {10}
    Net type:                      {11}
    '''.format(epochs, batch_size, lr, n_train, n_val, dice_lam, seg_lam, save_cp, device.type, img_scale, grid, net_type)
    )

  # weight_p, bias_p = [], []
  # for name, p in net.named_parameters():
  #   if 'bias' in name:
  #     bias_p += [p]
  #   else:
  #     weight_p += [p]
  # optimizer = optim.Adam([{'params': weight_p, 'weight_decay': 1e-5},
  #                           {'params': bias_p, 'weight_decay': 0}], lr=lr)

  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

  # scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)
  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

  ce_loss = nn.CrossEntropyLoss()
  dice_loss = DiceLoss(net.num_classes)

  global_step = save_ep * len(train_loader)
  max_iterations = epochs * len(train_loader)

  alpha = 0.25

  for epoch in range(save_ep, epochs):
    net.train()
    with tqdm(total=n_train, desc='Epoch {0}/{1}'.format(epoch+1, epochs), unit='img') as pbar:
      for batch in train_loader:
        img = batch['image']
        seg_lab = batch['seg_label']
        cls_lab = batch['cls_label']
        file_name = batch['name']
        img = img.to(device=device, dtype=torch.float32)
        seg_lab = seg_lab.to(device=device, dtype=torch.long)
        cls_lab = cls_lab.to(device=device, dtype=torch.long)
        # cls_lab = cls_lab.reshape(batch_size, -1, net.num_classes)

        if net_type == 'join':
          cls_pred, seg_pred = net(img)
          # cls_pred = cls_pred.reshape(batch_size, -1, net.num_classes)
          seg_loss = ce_loss(seg_pred, seg_lab[:].argmax(dim=1).long()) + dice_lam * dice_loss(seg_pred, seg_lab, softmax=True)
          cls_loss = ce_loss(cls_pred, cls_lab[:].argmax(dim=1).long())
          # cls_loss = sigmoid_focal_loss(inputs=cls_pred, targets=cls_lab[:].float(), alpha=alpha, reduction='mean')
          loss = seg_lam * seg_loss + cls_loss
          writer.add_scalar('cls_loss/train', cls_loss.detach(), global_step)
          writer.add_scalar('seg_loss/train', seg_loss.detach(), global_step)
          writer.add_scalar('join_loss/train', loss.detach(), global_step)
        if net_type == 'class':
          cls_pred = net(img)
          # cls_pred = cls_pred.reshape(batch_size, -1, net.num_classes)
          cls_loss = ce_loss(cls_pred, cls_lab[:].argmax(dim=1).long())
          # cls_loss = sigmoid_focal_loss(inputs=cls_pred, targets=cls_lab[:].float(), alpha=alpha, reduction='mean')
          loss = cls_loss
          writer.add_scalar('cls_loss/train', loss.detach(), global_step)
        if net_type == 'segment':
          seg_pred = net(img)
          seg_loss = ce_loss(seg_pred, seg_lab[:].argmax(dim=1).long()) + dice_lam * dice_loss(seg_pred, seg_lab)
          loss = seg_loss
          writer.add_scalar('seg_loss/train', loss.detach(), global_step)

        pbar.set_postfix(**{'loss (batch)': loss.detach()})

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        # learning rate
        lr_ = lr * (1.0 - global_step / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_
        writer.add_scalar('learning rate/lr', lr_, global_step)

        pbar.update(batch_size)
        global_step += 1

    if net_type == 'join':
      cls_result_patch, cls_result_slice, tumor_dice, liver_dice, consistency_score = eval_single_volume(net, net_type, val_loader, device, writer, epoch+1)
      logging.info(
        '''
        Live dice:         {0}
        Tumor dice:        {1}
        Patch accuracy:    {2}
        Slice accuracy:    {3}
        consistency_score: {4}
        '''.format(liver_dice, tumor_dice, cls_result_patch['acc'], cls_result_slice['acc'], consistency_score)
      )
    if net_type == 'class':
      cls_result_patch, cls_result_slice = eval_single_volume(net, net_type, val_loader, device, writer, epoch+1)
      logging.info(
        '''
        Patch accuracy: {0}
        Slice accuracy: {1}
        '''.format(cls_result_patch['acc'], cls_result_slice['acc'])
      )
    if net_type == 'segment':

      tumor_dice, liver_dice = eval_single_volume(net, net_type, val_loader, device, writer, epoch+1)
      logging.info(
        '''
        Live dice:      {0}
        Tumor dice:     {1}
        '''.format(liver_dice, tumor_dice)
      )


    if save_cp:
      try:
        os.mkdir(checkpoint_dir)
        logging.info('Create checkpoint directory')
      except OSError:
        pass
      torch.save(
        net.state_dict(),
        checkpoint_dir + 'CP_epoch{0}.path'.format(epoch + 1)
      )
      logging.info('Checkpont {0} saved !'.format(epoch +1))
  writer.close()
  return "Finish"

def get_args():

    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epochs', type=int, default=40, dest='epochs')
    parser.add_argument('--batch-size', type=int, default=12, dest='batchsize') # could be modified with 12 or 8
    parser.add_argument('--learning-rate', type=float, default=0.01, dest='lr')
    parser.add_argument('--load', type=str, default=False, dest='load')         # could be modified with the path of .path file
    parser.add_argument('--net_type', type=str, default='join', dest='net_type') # could be modified with class and segmentation
    parser.add_argument('--dice-lambda', type = float, default=1, dest='dice_lam')
    parser.add_argument('--seg-lambda', type = float, default=0.5, dest='seg_lam')
    parser.add_argument('--scale',  type=float, default=1, dest='img_scale')
    parser.add_argument('--grid',  type=float, default=7, dest='grid') # grid can only equal to 1, 3, 5, 7, re-run cls_lab_eng.py before you want to change the grid
    parser.add_argument('--residual',  type=bool, default=True, dest='residual')
    parser.add_argument('--save-cp',  type=bool, default=True, dest='save_cp')
    parser.add_argument('--save-ep',  type=int, default=0, dest='save_ep') # saved epochs number
    parser.add_argument('--seed',  type=int, default=22, dest='seed') # random seed

    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.net_type != 'join':
      assert args.residual is not True, "residual is designed only for join!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {0}'.format(device))

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

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

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {0}'.format(args.load))

    net.to(device=device)

    try:
        train_net(
          net=net,
          save_ep=args.save_ep,
          net_type=args.net_type,
          device=device,
          epochs=args.epochs,
          batch_size=args.batchsize,
          lr=args.lr,
          dice_lam=args.dice_lam,
          seg_lam=args.seg_lam,
          grid=args.grid,
          img_scale=args.img_scale,
          save_cp=args.save_cp,
          seed=args.seed
          )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

