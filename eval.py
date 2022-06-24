import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_loss import DiceLoss
from utils.coarse_score import coarse



def eval_net(net, loader, device, writer, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    lab_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_dice = 0
    tot_coarse = 0
    global_step = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=True) as pbar:
        for batch in loader:

            global_step += 1

            img, lab = batch['image'], batch['label']
            img = img.to(device=device, dtype=torch.float32)
            lab = lab.to(device=device, dtype=lab_type)

            with torch.no_grad():
                mask_pred = net(img)

            mask_pred = torch.sigmoid(mask_pred) > 0.5
            dice_score = DiceLoss.dice_coeff(mask_pred, lab).item()

            tot_dice += dice_score
            # tot_coarse += coarse(mask_pred, lab, writer, epoch, global_step).item()

            mask = torch.cat((lab, mask_pred), dim=2)
            if writer is not None:
                writer.add_images('masks/test_epoch{0}/{1}_{2}'.format(epoch, global_step, dice_score), mask, epoch)

            pbar.update(img.shape[0])

    net.train()

    return tot_dice / n_val               #, tot_coarse / n_val
