import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch, save_img, save_img1
import torch.nn.functional as F
import random
import cv2
from PIL import  Image
from torchvision import transforms
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################



class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        

        # input_dir = 'train_A'
        # mask_dir = 'train_D'
        input_dir = 'input'



        

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        # mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))




        self.out_path2 = os.path.join(rgb_dir, input_dir)
        # self.out_path3 = os.path.join(rgb_dir, mask_dir)



        
          # 文件路径保存在列表中
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        # self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]




        self.tar_size = len(self.noisy_filenames)  # get the size of target


    def __len__(self):
        return self.tar_size



    def __getitem__(self,index):


        tar_index   = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # mask = load_img(self.mask_filenames[tar_index])
        # mask = torch.from_numpy(np.float32(mask))




        # noisy = noisy.permute(2,0,1)
        # mask = mask.permute(2, 0, 1)


        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        # mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]





        # apply_trans = transforms_aug[random.getrandbits(3)]
        #
        #
        # noisy = getattr(augment, apply_trans)(noisy)
        # mask = getattr(augment, apply_trans)(mask)






        return noisy, noisy_filename











##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform


        # input_dir = 'test_A1'
        # mask_dir = 'test_B1'
        input_dir = 'input'

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        # mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))



        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        # self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]


        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        


        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # mask = load_img(self.mask_filenames[tar_index])
        # mask = torch.from_numpy(np.float32(mask))


        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
       # mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]









        return noisy,  noisy_filename
