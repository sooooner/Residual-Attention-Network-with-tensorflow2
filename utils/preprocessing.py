import tensorflow as tf
import numpy as np

def padding(img):
    return np.pad(img, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant')

def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def crop_generator(img):
    img = padding(img)
    return random_crop(img, (32, 32))