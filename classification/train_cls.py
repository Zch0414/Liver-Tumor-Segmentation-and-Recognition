import sys
sys.path.append("..")
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from eval_cls import *
from unet.resnet import ResNet, CatResNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_cls import BasicDataset
from torch.utils.data import DataLoader, random_split

train_path_dic = {
    'image': '../data_cls/train/image/',
    'label': '../data_cls/train/label/'
}
val_path_dic = {
    'image': '../data_cls/test/image/',
    'label': '../data_cls/test/label/'
}
dir_checkpoint = './checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.01,
              save_cp=True,
              name = 'resnet18',
              GOP=3):
    dataset_train = BasicDataset(train_path_dic, trans=True)
    dataset_val = BasicDataset(val_path_dic, trans=False)
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='_CLS_{0}_LR_{1}_BS_{2}_GOP_{3}'.format(name, lr, batch_size, GOP))
    global_step = 0

    logging.info('''Starting training:
               Epochs:          {0}
               Batch size:      {1}
               Learning rate:   {2}
               Training size:   {3}
               Validation size: {4}
               Checkpoints:     {5}
               Module Name      {6}
               Device:          {7}
           '''.format(epochs, batch_size, lr, n_train, n_val, save_cp, name, device.type))

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        epoch_step = 0
        with tqdm(total=n_train, desc='Epoch {0}/{1}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                img = batch['image'].to(device=device, dtype=torch.float32)
                lab = batch['label'].to(device=device, dtype=torch.float32)
                pred = net(img)

                loss = criterion(pred, lab)
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()
                epoch_step += 1

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(img.shape[0])
                global_step += 1

        fpr, tpr, _auc, acc = eval_net(net, val_loader, device)

        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        logging.info('Validation accuracy: {}'.format(acc))
        writer.add_scalar('Accuracy/test', acc, global_step)
        writer.add_scalar('FPR/test', fpr, global_step)
        writer.add_scalar('TPR/test', tpr, global_step)
        writer.add_scalar('AUC/test', _auc, global_step)
        writer.add_scalar('Loss/train_global', epoch_loss / epoch_step, epoch+1)

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
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gop', dest='GOP', type= int, nargs='?', default=1,
                        help='The group of picture')
    parser.add_argument('-n', '--net', dest='NET', type=str, nargs='?', default='resnet34',
                        help='The name of the candidate net')
    parser.add_argument('-c', '--cat', dest='CAT', type=bool, nargs='?', default=False,
                        help='Whether use the concatenate data and network')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {0}'.format(device))

    if args.CAT:
        net = CatResNet(args.NET, GOP=args.GOP)
    else:
        net = ResNet(args.NET, GOP=args.GOP)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {0}'.format(args.load))
    net.to(device=device)

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            name=args.NET,
            GOP=args.GOP
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
