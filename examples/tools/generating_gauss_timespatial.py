import numpy as np
from matplotlib import pyplot as plt


# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=2, sy=2):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def get_gauss_timespatial(image_size=30, frames_per_cycle=20, cycle_num=5):
    """
    Function for creating numpy array with moving gauss distributed circle
    To imitate spatiotemporal data
    """
    x = np.linspace(-3, 3, image_size)
    y = np.linspace(-3, 3, image_size)
    x, y = np.meshgrid(x, y)

    x_track = np.linspace(-6, 6, frames_per_cycle // 2)
    x_track = np.append(x_track, x_track[::-1])
    y_track = np.linspace(-6, 6, frames_per_cycle // 2)
    y_track = np.append(y_track, y_track[::-1])

    timeseries = []

    for n in range(cycle_num):
        for t in range(frames_per_cycle):
            z = gaus2d(x, y, mx=x_track[t], my=y_track[t])
            z = z / np.max(z)
            timeseries.append(z)

    return np.array(timeseries)


ts = get_gauss_timespatial()

# visualization of each timestep
for f in range(ts.shape[0]):
    plt.imshow(ts[f])
    plt.title(f'Frame number = {f}')
    plt.show()

# slice timeseries in point
ts_1d = ts[:, 5, 10]
plt.plot(ts_1d)
plt.show()