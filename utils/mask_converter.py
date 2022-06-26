import argparse
import cv2
import os
import numpy as np
from test_enhance_single_unalign import check_label, LABELS


def str2pixel(s):
    s = s[1:-1]
    s = s.split(',')
    rgb = [int(i) for i in s]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", type=str, help="rgb value of face in the input image")
    parser.add_argument("--background", type=str, help="rgb value of background in the input image")
    parser.add_argument("--left_eye", type=str, help="rgb value of left eye in the input image")
    parser.add_argument("--right_eye", type=str, help="rgb value of right eye in the input image")
    parser.add_argument("--nose", type=str, help="rgb value of nose in the input image")
    parser.add_argument("--lower_lip", type=str, help="rgb value of lower lip in the input image")
    parser.add_argument("--upperlip", type=str, help="rgb value of the upper lip in the input image")
    parser.add_argument("--neck", type=str, help="rgb value of the neck in the input image")
    parser.add_argument("--body", type=str, help="rgb value of the body in the input image")
    parser.add_argument("input_dir", type='str', help='path of the input image')
    parser.add_argument("output_dir", type=str, help="path of the output dir")
    parser = parser.parse_args()

    pixels = {
        'face': np.array(str2pixel(parser.face)),
        'left_eye': np.array(str2pixel(parser.face)),
        'right_eye': np.array(str2pixel(parser.face)),
        'nose': np.array(str2pixel(parser.face)),
        'upper_lip': np.array(str2pixel(parser.face)),
        'lower_lip': np.array(str2pixel(parser.face)),
        'neck': np.array(str2pixel(parser.face)),
        'body': np.array(str2pixel(parser.face)),
        'background': np.array(str2pixel(parser.face)),
    }
    img = cv2.imread(parser.input_dir)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            label = check_label(img[i][j], pixels)
            img[i][j][0] = LABELS[label][0]
            img[i][j][1] = LABELS[label][1]
            img[i][j][2] = LABELS[label][2]
    cv2.imwrite(parser.output_dir, img)


if __name__ == '__main__':
    main()
