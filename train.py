import argparse
import logging
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from eval import eval_net
from unet.unet_model import UNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.dice_loss import DiceLoss
from torch.utils.data import DataLoader, random_split

# I made some shortcuts when I coded the dataset.py.
# Therefore, make sure your directory contains the last '/'.

train_path_dic = {
    'image': './data_train/image/',
    'label': './data_train/label/'
}
val_path_dic = {
    'image': './data_test/image/',
    'label': './data_test/label/'
}
dir_checkpoint = './checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              _lambda = 0,
              img_scale=1):

    dataset_train = BasicDataset(train_path_dic, img_scale, trans=True)
    dataset_val = BasicDataset(val_path_dic, img_scale, trans=False)
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='LR_{0}_BS_{1}_LAMBDA_{2}'.format(lr, batch_size, _lambda))
    global_step = 0

    logging.info('''Starting training:
            Epochs:          {0}
            Batch size:      {1}
            Learning rate:   {2}
            Training size:   {3}
            Validation size: {4}
            Lambda value:    {5}
            Checkpoints:     {6}
            Device:          {7}
            Images scaling:  {8}
        '''.format(epochs, batch_size, lr, n_train, n_val, _lambda, save_cp, device.type, img_scale))

    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    # #optimizer = optim.Adam([{'params': weight_p, 'weight_decay': 1e-5},
    #                         {'params': bias_p, 'weight_decay': 0}], lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if net.n_classes == 1:
        ce_loss = nn.BCEWithLogitsLoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(net.n_classes)

    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc='Epoch {0}/{1}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                img = batch['image']
                lab = batch['label']
                assert img.shape[1] == net.n_channels, "Network has been defined with {0} input channels'format(), \
                          but loaded images have {0} channels. Please check that, \
                          the images are loaded correctly.".format(net.n_channels, img.shape[1])
                img = img.to(device=device, dtype=torch.float32)
                lab_type = torch.float32 if net.n_classes == 1 else torch.long
                lab = lab.to(device=device, dtype=lab_type)
                masks_pred = net(img)

                loss_ce = ce_loss(masks_pred, lab)
                loss_dice = dice_loss(masks_pred, lab)
                loss = (1 - _lambda) * loss_ce + _lambda * loss_dice

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()


                pbar.update(img.shape[0])

                global_step += 1


        val_score = eval_net(net, val_loader, device, writer, epoch+1)

        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Validation Dice Coeff: {}'.format(val_score))
        writer.add_scalar('Dice/test', val_score, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP_epoch{0}.pth'.format(epoch + 1))
            logging.info('Checkpoint {0} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--lambda', metavar='L', type = float, nargs='?', default=0.1,
                        help='lambda parameter of the dice loss', dest='_lambda')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')


    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {0}'.format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    logging.info('Network:\n'
                 '\t{0} input channels\n'
                 '\t{1} output channels (classes)\n'
                 '\t{2} upscaling'.format(net.n_channels, net.n_classes,
                                          "Bilinear" if net.bilinear else "Transposed conv"))

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {0}'.format(args.load))

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  _lambda=args._lambda,
                  img_scale=args.scale,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
