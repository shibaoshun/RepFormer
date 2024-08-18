"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
#from DGUNet import DGUNet
from sd_fsas_net import SDNet

from skimage import img_as_ubyte
from pdb import set_trace as stx
from data_RGB import get_training_data, get_validation_data, get_test_data_fine
parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='./rain100L/train', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./resultstishi/SDNet-train/', type=str, help='Directory for results')
parser.add_argument('--weights', default='E:\Maw\stride-ssh\stride-ssh\checkpoints\Deraining\models\StripeAwareNet\model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
parser.add_argument('--dim', type=int, default=256, help='')
parser.add_argument('--head_dim', type=int, default=64, help='')
parser.add_argument('--bias', type=str, default=False, help='')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# model_restoration = DGUNet()
model_restoration = utils.get_test(args)
utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

#datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
datasets = ['Rain100L']

def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def test_x8(model, L):
    E_list = [model(augment_img_tensor(L, mode=i))[0] for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

for dataset in datasets:
    # rgb_dir_test = os.path.join(args.input_dir,dataset)
    # print(rgb_dir_test)
    rgb_dir_test = args.input_dir
    print(rgb_dir_test)
    test_dataset = get_test_data_fine(rgb_dir_test, img_options={})
    print(test_dataset)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    print(test_loader)


    result_dir  = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)




    img_multiple_of = 8 * 8
    with torch.no_grad():
        rmse_val = []
        psnr_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            

            input_1    = data_test[0].cuda()
            target    = data_test[1].cuda()
            mask = data_test[2].cuda()
            filenames = data_test[3]
            # Pad the input if not_multiple_of win_size * 8
            height, width = input_1.shape[2], input_1.shape[3]
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_1, (0, padw, 0, padh), 'reflect')
            mask = F.pad(mask, (0, padw, 0, padh), 'reflect')
            restored = model_restoration(input_,mask)

#             restored = test_x8(model_restoration,input_)#model_restoration(input_)[0]


            restored = torch.clamp(restored, 0, 1)
            input_1 = torch.clamp(input_1, 0, 1)
            #            restored = torch.clamp(restored[0],0,1)
            restored = restored[:, :, :height, :width]
            for res, tar in zip( restored,input_1 - target):
                rmse = torch.sqrt(torch.mean(((res - target)) ** 2))
                rmse_val.append(rmse)
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

            # restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = restored.permute(0, 2, 3, 1).cpu().detach()
            y = len(restored)
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img) ## 进行生成DRCD所需的估计图像时，这里生成修改成三通道，在函数里改

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        rmse = torch.stack(rmse_val).mean().item()
        print("PSNR: %f, RMSE: %f " %(psnr_val_rgb, rmse))
        psnr_val_rgb1 = sum(psnr_val_rgb) / len(test_dataset)
        rmse1 = sum(rmse_val) / len(test_dataset)
        print("PSNR1: %f, RMSE1: %f " % (psnr_val_rgb1, rmse1))
