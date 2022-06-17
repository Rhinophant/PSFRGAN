import cv2
import dlib
import os
import numpy as np
from skimage import io
import torch
from tqdm import tqdm

from utils import utils
from models import create_model
from options.test_options import TestOptions


def save_imgs(img, save_dir, filename):
    save_path = os.path.join(save_dir, filename)
    io.imsave(save_path, img.astype(np.uint8))


def generate_masks(opt):
    file_names = os.listdir(opt.test_img_path)
    model = create_model(opt)
    model.load_pretrain_models()
    model.netP.to(opt.device)
    for f in tqdm(file_names):
        img = dlib.load_rgb_image(os.path.join(opt.test_img_path, f))
        if img.shape != (512, 512, 3):
            img = cv2.resize(img, (512, 512))
        with torch.no_grad():
            lq_tensor = torch.tensor(img.transpose(2, 0, 1)) / 255. * 2 - 1
            lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)
            parse_map, _ = model.netP(lq_tensor)
            parse_map_onehot = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
        parse_map = utils.color_parse_map(parse_map_onehot)[0]
        save_imgs(parse_map, opt.results_dir, f)


if __name__ == '__main__':
    opt = TestOptions().parse()
    generate_masks(opt)

