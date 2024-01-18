import argparse
import logging
import os
from cv2 import split
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from sklearn.metrics import accuracy_score

def predict_img(net,
                full_img,
                device,
                scale_factor=1500,
                out_threshold=0.5):
    net.eval()
    #print('prep:::',BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    #img = torch.from_numpy(np.array(full_img))
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    #print('net shape:',img.shape)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return full_mask.argmax(dim=0).numpy()

def predict_img_with_prob(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    if img.shape[1] == 1:
        img = img.repeat([1, 3, 1, 1])
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0][1]
        else:
            probs = torch.sigmoid(output)[0]

        full_mask = probs.cpu().detach().numpy()
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/mnt/disk1/luo/unet/unet/checkpoints/checkpoint_epoch30.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--test', default='./test', type=str, help='Filenames of input images')
    parser.add_argument('--output', default='./output/', type=str, help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1500,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        tmp = mask * 255
        tmp[tmp > 0] = 255
        return Image.fromarray(tmp.astype(np.uint8))
    elif mask.ndim == 3:
        tmp = np.argmax(mask, axis=0) * 255 / mask.shape[0]
        tmp[tmp > 0] = 255
        return Image.fromarray(tmp.astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output, exist_ok=True)


    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Loading model {args.model}')
    #print(f'Using device {device}')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')
    img_paths = [str(p) for p in list(Path(args.test).joinpath("imgs").glob("*.jpg"))]
    mask_paths = [str(p) for p in list(Path(args.test).joinpath("masks").glob("*.jpg"))]
    #print(f'img:{img_paths}')
    #print(f'mask_paths:{mask_paths}')
    mean_acc = 0
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        name = img_path.split('/')[-1]
        #print(f'name:{img_path}')
        img_path = './'+img_path
        img = Image.open(img_path)
        #print(f'name11111:{img_path}')
        mask_path = './'+mask_path
        true_mask = np.array(Image.open(mask_path))
        #print(f'shape::{true_mask.shape}')
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        #print(f'mask_pred::{mask.shape}')
        if len(true_mask.shape) == 3:
            true_mask = true_mask[:, :, 0]

        acc = accuracy_score(np.array(true_mask > 0, dtype=np.int32).reshape(-1), mask.reshape(-1))
        print(f"图像名: {name}\t准确率: {acc}")
        mean_acc += acc / len(img_paths)

        if not args.no_save:
            result = mask_to_image(mask)
            # result.save(os.path.join(args.output, f"{name}.png"))
            res_arr = np.array(result)
            true_mask = true_mask[..., np.newaxis].repeat(3, axis=2)
            res_arr = res_arr[..., np.newaxis].repeat(3, axis=2)

            merge = np.hstack([np.array(img), true_mask, res_arr])
            Image.fromarray(merge).save(os.path.join(args.output, f"{name}_{acc}.png"))

        if args.viz:
            logging.info(f'Visualizing results for image {img_path}, close to continue...')
            plot_img_and_mask(img, mask)
    print(f"平均准确率: {mean_acc}")
