import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    acc = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(
            0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                
                acc += accuracy_score(mask_true.argmax(dim=1).reshape(-1).detach().cpu().numpy(),
                                      mask_pred.argmax(dim=1).reshape(-1).detach().cpu().numpy())

                true_img = (mask_true.argmax(dim=1).squeeze().detach().cpu().numpy() * 255).astype(int)
                pred_img = (mask_pred.argmax(dim=1).squeeze().detach().cpu().numpy() * 255).astype(int)
                #img = np.hstack([true_img, pred_img])
                img = pred_img
                #print('true:{}',true_img.shape)
                #print('pred:{}',pred_img.shape)
                cv2.imwrite(f'./test_prog/{acc}.png', img)
                

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, acc / num_val_batches
