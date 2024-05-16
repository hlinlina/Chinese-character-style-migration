from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import numpy as np
from io import BytesIO
from skimage.transform import resize


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def load_test_data(image_path, size=256):
    img = imageio.imread(image_path)
    img = resize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img


def preprocessing(x):
    x = x / 127.5 - 1
    return x


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = imageio.imread(img).astype(np.float)
    #print("\n!!!!!!!!!!!~~~~~~~~~: ",mat.shape)
    side = int(mat.shape[1] / 3)
    assert side * 3 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:side + side]  # source
    img_C = mat[:, side + side:]

    return img_A, img_B, img_C


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    h, w, _ = img.shape
    enlarged = resize(img, [nw, nh])
    return enlarged[shift_x:shift_x + h, shift_y:shift_y + w, :]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    imageio.imwrite(img_path, concated)


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [resize(imageio.imread(f), (int(0.33*f.shape[0]), int(0.33*f.shape[1])), preserve_range=True) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file