
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchcnnbuilder.models import ForecasterBase
def prapare_dataset(move: str, persons: np.ndarray, frames_num: int):
    root = f'D:/KTH/{move}'
    target_files =[f'person{str(p).zfill(2)}_{move}_d{n}_uncomp.avi' for p in persons for n in range(1, 4)]
    dataset = []
    for i, file in enumerate(target_files):
        print(f'load file {i}/{len(target_files)}')
        vid = cv2.VideoCapture(f'{root}/{file}')
        frames = []
        while len(frames) < frames_num:
            _, arr = vid.read()
            arr = np.mean(arr, axis=2) / 255
            frames.append(arr)
        dataset.append(frames)
    dataset = np.array(dataset)
    return dataset

batch_size = 100

start_frame = 20

test_series = prapare_dataset('walking', np.arange(17, 25), start_frame+50)[:, start_frame:, :, :]
test_series = resize(test_series, (test_series.shape[0], test_series.shape[1],  128, 128))

device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=[128, 128],
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())
model.load_state_dict(torch.load(f'models/KTH_19000.pt', weights_only=True))
model.eval()
model = model.to(device)


pred_times = 4 # предсказываем столько раз по 10 кадров
for s in range(5):
    full_prediction = []
    input = torch.Tensor(test_series[s, :10, :, :]).to(device)
    for t in range(pred_times):
        input = model(input)
        full_prediction.extend(input.detach().cpu().numpy().tolist())

    fig, axs = plt.subplots(2, len(full_prediction), figsize=(len(full_prediction), 3))
    for i in range(len(full_prediction)):
        axs[0][i].imshow(full_prediction[i], cmap='Greys_r')
        axs[0][i].set_title(f't={11 + i}')
        axs[1][i].imshow(test_series[s, i+10, :, :], cmap='Greys_r')

        axs[0][i].axes.xaxis.set_ticks([])
        axs[1][i].axes.xaxis.set_ticks([])
        axs[0][i].axes.yaxis.set_ticks([])
        axs[1][i].axes.yaxis.set_ticks([])

    plt.suptitle(f'Sample {s}')
    plt.tight_layout()
    plt.show()

