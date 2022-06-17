import argparse
import numpy
import numpy as np
import math
import cv2
import torch
import pytorch_ssim
from torch.autograd import Variable
import os


def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', help='direct of test images')
    parser.add_argument('--ground_truth_dir', help='direct of ground truth images')
    parser = parser.parse_args()

    original_pics = []
    contrast_pics = []

    for root, dirs, files in os.walk(parser.test_dir):
        for i in files:
            original_pics.append(cv2.imread(root + '\\\\' + i))
    for root, dirs, files in os.walk(parser.ground_truth_dir):
        for i in files:
            contrast_pics.append(cv2.imread(root + '\\\\' + i))

    for i in range(len(contrast_pics)):
        if contrast_pics[i].shape != (512, 512):
            contrast_pics[i] = cv2.resize(contrast_pics[i], (512, 512))
    for i in range(len(original_pics)):
        if original_pics[i].shape != (512, 512):
            original_pics[i] = cv2.resize(original_pics[i], (512, 512))

    psnr_values = []
    ssim_values = []

    for i in range(len(original_pics)):
        psnr_values.append(psnr(original_pics[i], contrast_pics[i]))
        ssim_values.append(ssim(original_pics[i], contrast_pics[i]))

    print(psnr_values)
    print(ssim_values)
    print("PSNR average: " + str(np.average(psnr_values)))
    print("PSNR variance: " + str(np.var(psnr_values)))
    print("SSIM average: " + str(np.average(ssim_values)))
    print("SSIM variance: " + str(np.var(ssim_values)))
