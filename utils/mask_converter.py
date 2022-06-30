import argparse
import cv2
import numpy as np
import utils


def str2pixel(s):
    if s is None:
        return [-1, -1, -1]
    s = s[1:-1]
    s = s.split(',')
    rgb = [int(i) for i in s]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", type=str, help="rgb value of face in the input image, (r,g,b)")
    parser.add_argument("--background", type=str, help="rgb value of background in the input image, (r,g,b)")
    parser.add_argument("--left_eye", type=str, help="rgb value of left eye in the input image, (r,g,b)")
    parser.add_argument("--right_eye", type=str, help="rgb value of right eye in the input image, (r,g,b)")
    parser.add_argument("--nose", type=str, help="rgb value of nose in the input image, (r,g,b)")
    parser.add_argument("--lower_lip", type=str, help="rgb value of lower lip in the input image, (r,g,b)")
    parser.add_argument("--upper_lip", type=str, help="rgb value of the upper lip in the input image, (r,g,b)")
    parser.add_argument("--neck", type=str, help="rgb value of the neck in the input image, (r,g,b)")
    parser.add_argument("--body", type=str, help="rgb value of the body in the input image, (r,g,b)")
    parser.add_argument("--input_dir", type=str, help='path of the input image')
    parser.add_argument("--output_dir", type=str, help="path of the output dir")
    parser = parser.parse_args()

    pixels = {
        'face': np.array(str2pixel(parser.face)),
        'left_eye': np.array(str2pixel(parser.left_eye)),
        'right_eye': np.array(str2pixel(parser.right_eye)),
        'nose': np.array(str2pixel(parser.nose)),
        'upper_lip': np.array(str2pixel(parser.upper_lip)),
        'lower_lip': np.array(str2pixel(parser.lower_lip)),
        'neck': np.array(str2pixel(parser.neck)),
        'body': np.array(str2pixel(parser.body)),
        'background': np.array(str2pixel(parser.background)),
    }
    img = cv2.imread(parser.input_dir)
    b = img[:, :, 0].copy().T
    g = img[:, :, 1].copy().T
    r = img[:, :, 2].copy().T
    img = np.array([r, g, b]).transpose((2, 1, 0))
    h, w, _ = img.shape

    for i in range(h):
        for j in range(w):
            label = utils.check_label(img[i][j], pixels)
            img[i][j][0] = utils.LABELS[label][0]
            img[i][j][1] = utils.LABELS[label][1]
            img[i][j][2] = utils.LABELS[label][2]
    cv2.imwrite(parser.output_dir, img)
    print("File written to " + parser.output_dir)


if __name__ == '__main__':
    main()
