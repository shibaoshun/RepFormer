import os
from config import Config

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from rsen import RSENet

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import time
import datetime

######### Logs dir ###########
session = opt.MODEL.SESSION
log_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'log', session)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
now = datetime.datetime.now().isoformat().replace(':', '-')
logname = os.path.join(log_dir, now +'.txt')
print("Now time is : ", datetime.datetime.now().isoformat())

######### Set Seeds ###########  迭代优化（根据上一次的值计算下一次）
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.cuda()
with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

total_params = sum(p.numel() for p in model_restoration.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model_restoration.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.AdamW(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
# criterion_vgg = VGGLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=0,# 16
                          drop_last=False, pin_memory=True)# True# opt.OPTIM.BATCH_SIZE

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False,
                        pin_memory=True)# 16

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()

    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()

        stripe = model_restoration(input_,mask)

        # Compute loss at each stage
        # loss_st = 0

        loss = F.mse_loss(input_ - stripe,target) # 估计条纹损失函数

        #loss_vgg = criterion_vgg(restored,target) * 0.01   #

        # requires_grad = True
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        r"""
        loss_char = np.sum([criterion_char(restored[j], target) for j in range(len(restored))])
        loss_edge = np.sum([criterion_edge(restored[j], target) for j in range(len(restored))])
        loss = (loss_char) + (0.05 * loss_edge)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        """




    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            mask = data_val[2].cuda()

            with torch.no_grad():
                stripe = model_restoration(input_,mask)
            #restored = restored[0]

            for res, tar in zip(input_ - stripe, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
        with open(logname, 'a') as f:
            f.write("[epoch %d PSNR: %.4f\t ---- best_epoch %d Best_PSNR %.4f] " \
                    % (epoch, psnr_val_rgb, best_epoch, best_psnr) + '\n')
        """"
                if os.path.exists(log_dir):
            checkpoint = torch.load(log_dir)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')

        for epoch in range(start_epoch + 1, epochs):
            train(model, train_load, epoch)
            test(model, test_load)
            """

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,epoch_loss,scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
