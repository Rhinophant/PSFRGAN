'''
This script enhance all faces in one image with PSFR-GAN and paste it back to the original place.
'''
import dlib
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform as trans
from skimage import io
import time

import torch
from utils import utils
from options.test_options import TestOptions
from models import create_model

LABELS = {
        'face': np.array([0, 128, 0]),
        'left_eye': np.array([0, 0, 128]),
        'right_eye': np.array([64, 0, 0]),
        'nose': np.array([128, 128, 128]),
        'upper_lip': np.array([192, 0, 0]),
        'lower_lip': np.array([128, 0, 128]),
        'neck': np.array([0, 128, 128]),
        'body': np.array([128, 0, 0]),
        'background': np.array([0, 0, 0]),
    }


def check_label(pixel, labels):
    for key, value in enumerate(labels):
        if value[0] == pixel[0] and value[1] == pixel[1] and value[2] == pixel[2]:
            return key
    return 'background'


def parsemap2tensor(path):
    position = {
        'face': 1,
        'left_eye': 5,
        'right_eye': 4,
        'nose': 2,
        'upper_lip': 11,
        'lower_lip': 12,
        'neck': 17,
        'body': 18,
        'background': 0,
    }
    img = cv2.imread(path)
    height, width = img.shape[0], img.shape[1]
    assert height == 512 and width == 512
    tensor = np.zeros((1, 19, 512, 512))
    for i in range(height):
        for j in range(width):
            label = check_label(img[i][j], LABELS)
            tensor[0][position[label]][i][j] = 1
    return torch.tensor(tensor, dtype=torch.float32)


def detect_and_align_faces(img, face_detector, lmk_predictor, template_path, template_scale=2, size_threshold=999):
    align_out_size = (512, 512)
    ref_points = np.load(template_path) / template_scale

    # Detect landmark points
    face_dets = face_detector(img, 1)
    assert len(face_dets) > 0, 'No faces detected'

    aligned_faces = []
    tform_params = []
    for det in face_dets:
        if isinstance(face_detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect  # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold:
            print('Face is too large')
            break
        landmark_points = lmk_predictor(img, rec)
        single_points = []
        for i in range(5):
            single_points.append([landmark_points.part(i).x, landmark_points.part(i).y])
        single_points = np.array(single_points)
        tform = trans.SimilarityTransform()
        tform.estimate(single_points, ref_points)
        tmp_face = trans.warp(img, tform.inverse, output_shape=align_out_size, order=3)
        aligned_faces.append(tmp_face * 255)
        tform_params.append(tform)
    return [aligned_faces, tform_params]


def def_models(opt):
    model = create_model(opt)
    model.load_pretrain_models()
    model.netP.to(opt.device)
    model.netG.to(opt.device)
    return model


def enhance_faces(LQ_faces, model, opt):
    hq_faces = []
    lq_parse_maps = []
    for lq_face in tqdm(LQ_faces):
        with torch.no_grad():
            lq_tensor = torch.tensor(lq_face.transpose(2, 0, 1)) / 255. * 2 - 1
            lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)
            parse_map, _ = model.netP(lq_tensor)
            if opt.manual_parse == 'True' and opt.manual_parse_map_dir != '':
                parse_map_onehot = parsemap2tensor(opt.manual_parse_map_dir)
            elif opt.manual_parse == 'True' and opt.manual_parse_map_dir == '':
                print('Manual parse map not specified, using automatic parsing map instead.')
                parse_map_onehot = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            else:
                parse_map_onehot = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            output_SR = model.netG(lq_tensor, parse_map_onehot)
        hq_faces.append(utils.tensor_to_img(output_SR))
        lq_parse_maps.append(utils.color_parse_map(parse_map_onehot)[0])
    return hq_faces, lq_parse_maps


def past_faces_back(img, hq_faces, tform_params, upscale=1):
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)
    for hq_img, tform in tqdm(zip(hq_faces, tform_params), total=len(hq_faces)):
        tform.params[0:2, 0:2] /= upscale
        back_img = trans.warp(hq_img / 255., tform, output_shape=[int(h * upscale), int(w * upscale)], order=3) * 255

        # blur mask to avoid border artifacts
        mask = (back_img == 0)
        mask = cv2.blur(mask.astype(np.float32), (5, 5))
        mask = (mask > 0)
        img = img * mask + (1 - mask) * back_img
    return img.astype(np.uint8)


def save_imgs(img_list, save_dir, img_names):
    for idx, img in enumerate(img_list):
        # cv2.imwrite(os.path.join(save_dir, '%s.jpg' % img_names[idx]), img)
        io.imsave(os.path.join(save_dir, '%s.jpg' % img_names[idx]), img.astype(np.uint8))


if __name__ == '__main__':
    start_time = time.clock()
    opt = TestOptions().parse()
    #  face_detector = dlib.get_frontal_face_detector()
    face_detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    lmk_predictor = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')
    template_path = './pretrain_models/FFHQ_template.npy'

    print('======> Loading images, crop and align faces.')
    img_path = opt.test_img_path
    imgs = []
    img_names = []
    if os.path.isdir(opt.test_img_path):
        for f in os.listdir(opt.test_img_path):
            try:
                img = dlib.load_rgb_image(os.path.join(opt.test_img_path, f))
            except RuntimeError:
                print('%s is not an image file.' % f)
                continue
            assert img.shape[0] == img.shape[1], 'Height and width of the image must be the same!'
            img = cv2.resize(img, (512, 512))
            imgs.append(img)
            img_names.append(f)
    else:
        imgs.append(dlib.load_rgb_image(opt.test_img_path))
        img_names.append(os.path.basename(opt.test_img_path))
    print('%d images found...' % len(imgs))
    aligned_faces = np.array(imgs)

    # Save aligned LQ faces
    save_lq_dir = os.path.join(opt.results_dir, 'LQ_faces')
    os.makedirs(save_lq_dir, exist_ok=True)
    print('======> Saving aligned LQ faces to', save_lq_dir)
    save_imgs(aligned_faces, save_lq_dir, img_names)

    # Load models, and enhance the faces
    enhance_model = def_models(opt)
    hq_faces, lq_parse_maps = enhance_faces(aligned_faces, enhance_model, opt)

    # Save LQ parsing maps and enhanced faces
    save_parse_dir = os.path.join(opt.results_dir, 'ParseMaps')
    save_hq_dir = os.path.join(opt.results_dir, 'HQ')
    os.makedirs(save_parse_dir, exist_ok=True)
    os.makedirs(save_hq_dir, exist_ok=True)
    print('======> Save parsing map and the enhanced faces.')
    save_imgs(lq_parse_maps, save_parse_dir, img_names)
    save_imgs(hq_faces, save_hq_dir, img_names)

    end_time = time.clock()
    print('Time used: %s seconds.' % (end_time - start_time))
