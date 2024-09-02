import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt


def get_anime_timeseries(rgb=False):
    with Image.open('media/anime_10f.gif') as im:
        array = []
        for frame in ImageSequence.Iterator(im):
            if rgb:
                im_data = frame.copy().convert('RGB').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 3)
            else:
                im_data = frame.copy().convert('L').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 1)
            array.append(im_array)
        array = np.array(array)
        '''plt.imshow(array[0, :, :, :], cmap='Greys_r')
        plt.colorbar()
        plt.show()'''
        array = 255/array
    return array


def get_cycled_data(array, cycles_num):
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4])
    return arr


def get_long_data():
    im = get_anime_timeseries()
    long_im = []
    for i in range(9):
        missing_matrix = np.linspace(im[i], im[i+1], 7)[:-1]
        long_im.extend(missing_matrix)
    long_im = np.array(long_im)[:52]
    return long_im

long_im = get_long_data()
full_sinthetic_timeseries = get_cycled_data(long_im, 10)

for im in full_sinthetic_timeseries:
    plt.imshow(im, cmap='Greys_r')
    plt.show()

