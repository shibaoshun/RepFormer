import os
from PIL import Image
from torchvision import transforms
import random
import torch
import numpy
import torchvision.utils as vutils
import cv2

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])# 90  k旋转次数
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2]) # 180
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2]) # 270
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)  # 翻转  以某一个维度为轴进行翻转
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2) # 先进行旋转再进行翻转
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
def main():
    input1_dir = './DATA/train/train_A1'  # 存放原始图片的文件夹路径
    input2_dir = './DATA/train/train_B1'
    input3_dir = './DATA/train/train_C1'
    input4_dir = './DATA/train/train_D1'

    output_dir_clean = './DATA/train/train_A'  # 存放所有操作后图片的文件夹路径
    output_dir_noisy = './DATA/train/train_B'
    output_dir_mask = './DATA/train/train_C'
    output_dir_mask2 = './DATA/train/train_D'

    os.makedirs(output_dir_clean, exist_ok=True)
    os.makedirs(output_dir_noisy, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_mask2, exist_ok=True)
    #遍历文件夹中的图片文件
    count = 1  # 文件名递增值
    augment   = Augment_RGB_torch()# 实例化
    transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

    for filename in os.listdir(input1_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 加载图片
            clean_path = os.path.join(input1_dir, filename)
            noisy_path = os.path.join(input2_dir, filename)
            mask_path = os.path.join(input3_dir, filename)
            mask2_path = os.path.join(input4_dir, filename)

            clean = Image.open(clean_path)
            noisy = Image.open(noisy_path)
            mask = Image.open(mask_path)
            mask2 = Image.open(mask2_path)

            # to_tensor = transforms.ToTensor()
            # clean = to_tensor(clean)


            aug_times = 4
            for m in range(aug_times):
                to_tensor = transforms.ToTensor()
                clean = to_tensor(clean)
                noisy = to_tensor(noisy)
                mask = to_tensor(mask)
                mask2 = to_tensor(mask2)

                apply_trans = transforms_aug[m]

                clean = getattr(augment, apply_trans)(clean)
                noisy = getattr(augment, apply_trans)(noisy)
                mask = getattr(augment, apply_trans)(mask)
                mask2 = getattr(augment, apply_trans)(mask2)

                output_clean = os.path.join(output_dir_clean, f"{count}.png")
                output_noisy = os.path.join(output_dir_noisy, f"{count}.png")
                output_mask = os.path.join(output_dir_mask, f"{count}.png")
                output_mask2 = os.path.join(output_dir_mask2, f"{count}.png")

                to_image = transforms.ToPILImage()
                clean = to_image(clean)
                noisy = to_image(noisy)
                mask = to_image(mask)
                mask2 = to_image(mask2)

                clean.save(output_clean)
                noisy.save(output_noisy)
                mask.save(output_mask)
                mask2.save(output_mask2)

                count += 1
if __name__ == "__main__":
    main()
# for filename in os.listdir(input_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         # 加载图片
#         image_path = os.path.join(input_dir, filename)
#         image = Image.open(image_path)
#         # 旋转90度
#         rotate_90_image = image.transpose(Image.ROTATE_90)
#         rotate_90_output_path = os.path.join(output_dir_all, f"{count}.jpg")
#         rotate_90_image.save(rotate_90_output_path)
#         count += 1
#         # 旋转180度
#         rotate_180_image = image.transpose(Image.ROTATE_180)
#         rotate_180_output_path = os.path.join(output_dir_all, f"{count}.jpg")
#         rotate_180_image.save(rotate_180_output_path)
#         count += 1
#         # 旋转270度
#         rotate_270_image = image.transpose(Image.ROTATE_270)
#         rotate_270_output_path = os.path.join(output_dir_all, f"{count}.jpg")
#         rotate_270_image.save(rotate_270_output_path)
#         count += 1
#         # 水平翻转
#         horizontal_flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
#         horizontal_flip_output_path = os.path.join(output_dir_all, f"{count}.jpg")
#         horizontal_flip_image.save(horizontal_flip_output_path)
#         count += 1
#         # 垂直翻转
#         vertical_flip_image = image.transpose(Image.FLIP_TOP_BOTTOM)
#         vertical_flip_output_path = os.path.join(output_dir_all, f"{count}.jpg")
#         vertical_flip_image.save(vertical_flip_output_path)
#         count += 1
