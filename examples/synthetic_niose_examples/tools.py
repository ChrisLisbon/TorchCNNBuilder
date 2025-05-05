import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Tuple, List
from logging import Logger
import logging


def synthetic_time_series(num_frames: int = 210,
                          matrix_size: int = 100,
                          square_size: int = 25,
                          interval: int = 10) -> Tuple[animation.ArtistAnimation, List[np.array]]:
    """
    Creating a synthetic 2d time series with a square moving in a circle

    :param num_frames: total number of frames (tensors in the 2d time series). Default: 210
    :param matrix_size: tensor sizes (input tensor). Default: 100
    :param square_size: square sizes (inside the input tensor). Default: 25
    :param interval: interval between frames in milliseconds. Default: 10
    :return:
    """
    matrix = np.ones((matrix_size, matrix_size))

    # generating a square in the picture (2-dimensional input tensor)
    # the square consists of zeros and the background consists of ones, because cmap='gray' makes the colors polar
    start_index = (matrix_size - square_size) // 2
    end_index = start_index + square_size
    matrix[start_index:end_index, start_index:end_index] = 0

    fig, ax = plt.subplots()

    # parameterization
    theta = np.linspace(0, 2*np.pi, num_frames)

    frames = []
    matrices = []
    for i in range(num_frames):

        # shifting using parameterization of trigonometric functions
        x_shift = int(matrix_size/2 + matrix_size/4 * np.cos(theta[i]))
        y_shift = int(matrix_size/2 + matrix_size/4 * np.sin(theta[i]))

        shifted_matrix = np.roll(matrix, x_shift - matrix_size//2, axis=1)
        shifted_matrix = np.roll(shifted_matrix, y_shift - matrix_size//2, axis=0)

        frames.append((ax.imshow(shifted_matrix, cmap='gray'),))
        matrices.append(1 - shifted_matrix)

    ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=interval)
    return ani, matrices


def save_gif(matrices: np.array,
             path: str,
             writer: str = 'pillow',
             fps: int = 30,
             interval: int = 10) -> None:
    """
    Saving matrices lika a .gif file

    :param matrices: array of frames
    :param path: path
    :param writer: writer of frames. Default: pillow
    :param fps: fps. Default: 30
    :param interval: interval between frames in milliseconds. Default: 10
    """
    plt.figure()
    fig, ax = plt.subplots()
    frames = list(map(lambda x: (ax.imshow(1-x, cmap='gray'),), matrices))
    ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=interval)
    ani.save(filename=f'{path}.gif', writer=writer, fps=fps)
    plt.close()


def IoU(X: torch.Tensor,
        Y: torch.Tensor,
        eps: float = 1e-8,
        threshold: float = 0.9):
    """
    Implementation of IoU metric (intersection / union)

    :param X: X tensor
    :param Y: Y tensor
    :param eps: smooth value to escape dividing by zero
    :param threshold: value to binarize tensors
    :return: IoU values in range of [0; 1]
    """
    X = X > threshold
    Y = Y > threshold

    dimensions = [i for i in range(0, len(X.shape))]

    intersection = (X & Y).float().sum(dimensions)
    union = (X | Y).float().sum(dimensions)
    iou = (intersection + eps) / (union + eps)

    return iou.mean()


def create_logger(log_file: str) -> Logger:
    """
    Create a logger object

    :param log_file: path to the file for logging
    :return logging.RootLogger: logger object
    """
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_print(logger, message) -> None:
    """
    Log and print at the same time

    :param logger:
    :param message:
    :return:
    """
    logger.info(message)
    print(message)


if __name__ == '__main__':
    ani, frames = synthetic_time_series()
    ani.save('time_series_animation.gif', writer='pillow', fps=30)