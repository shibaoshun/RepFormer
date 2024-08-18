import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))# 加入提示学习的时候这里需要输入sam提取的图片

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        mask_img = Image.open(mask_path)

        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
            msak_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        mask_img = TF.to_tensor(mask_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
        mask_img = mask_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            mask_img = mask_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            mask_img = mask_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
            mask_img = torch.rot90(mask_img, dims=(1, 2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            mask_img = torch.rot90(mask_img, dims=(1, 2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            mask_img = torch.rot90(mask_img.flip(1), dims=(1, 2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
            mask_img = torch.rot90(mask_img.flip(2), dims=(1, 2))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img,mask_img, filename

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)
        mask_img = Image.open(mask_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))
            mask_img = TF.center_crop(mask_img, (ps, ps))
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        mask_img = TF.to_tensor(mask_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img,mask_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename

class DataLoaderTest_fine(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest_fine, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(inp_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(inp_dir, 'target')))
        mask_files = sorted(os.listdir(os.path.join(inp_dir, 'mask')))

        self.inp_filenames = [os.path.join(inp_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(inp_dir, 'target', x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(inp_dir, 'mask', x) for x in mask_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        path_tar = self.tar_filenames[index]
        path_mask = self.mask_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp_img = Image.open(path_inp)
        tar_img = Image.open(path_tar)
        mask_img = Image.open(path_mask)

        inp = TF.to_tensor(inp_img)
        tar = TF.to_tensor(tar_img)
        mask = TF.to_tensor(mask_img)
        return inp,tar, mask,filename