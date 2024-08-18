import numpy as np
import skimage
import cv2
import argparse
import os
import utils
from utils.loader import get_training_data, get_validation_data
from torch.utils.data import DataLoader
from skimage.segmentation import find_boundaries
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint='./checkpoints/sam_vit_b_01ec64.pth')
sam.cuda(1)
mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.5)


def SAMAug(tI, mask_generator):
    masks = mask_generator.generate(tI)
    tI = skimage.img_as_float(tI)
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    BoundaryPrior=np.zeros((tI.shape[0],tI.shape[1]))
    for maskindex in range(len(masks)):
        thismask=masks[maskindex]['segmentation']
        stability_score = masks[maskindex]['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
        BoundaryPrior=BoundaryPrior+find_boundaries(thismask_,mode='thick')
        BoundaryPrior[np.where(BoundaryPrior>0)]=1
    tI[:,:,1] = tI[:,:,1]+SegPrior
    tI[:,:,2] = tI[:,:,2]+BoundaryPrior
    return BoundaryPrior

# image = cv2.imread("./data/Kvasir-SEG/images/1-3.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# output = SAMAug(image, mask_generator)
# img = output * 255
# cv2.imwrite("./data/Kvasir-SEG/images/88.png", img)
# cv2.imshow("image", output)
# cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='./Rain200L/train', help='path to train data')
    parser.add_argument('--test_dir', type=str, default='./Rain200L/test', help='path to train data')
    parser.add_argument('--train_workers', default=0, type=int, help='Directory for results')
    parser.add_argument('--result_dir', default='./rain200L/train/', type=str, help='Directory for results')
    parser.add_argument('--result2_dir', default='./yuxian', type=str, help='Directory for results')
    args = parser.parse_args()
    train_dataset = get_training_data(args.train_dir)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=args.train_workers, pin_memory=True)

    test_dataset = get_validation_data(args.test_dir)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=args.train_workers, pin_memory=False)



    for i, data in enumerate(test_loader, 0):

        input_ = data[0].cuda()
        # mask = data[1].cuda()
        # mask2 = data[2].cuda()
        filenames = data[1]
        input_= input_.squeeze(dim=0)
        # mask = mask.squeeze(dim=0)
        # mask2 = mask2.squeeze(dim=0)


        # y = input_ - mask
        y = input_
        y = y.cpu().numpy()
        y = y.astype(np.uint8)

        sam = SAMAug(y, mask_generator)



        sam = sam * 255
        utils.save_img1(sam, os.path.join(args.result2_dir, filenames[0]))

if __name__ == '__main__':
    main()