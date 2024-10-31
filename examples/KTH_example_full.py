import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchcnnbuilder.models import ForecasterBase


# https://arxiv.org/pdf/2308.09891
# resizing each image to 128 × 128 and using persons 1-16 for training and 17-25 for testing
# The models predict 10 frames from 10 observations at training time and 20 or 40 frames at inference time

moves = [ 'handwaving', 'walking', 'running', 'jogging', 'handclapping', 'boxing']

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

epochs = 20000
batch_size = 100

ds = []
for m in moves:
    train_series = prapare_dataset(m, np.arange(1, 17), 40)[:, 20:, :, :]
    train_series = resize(train_series, (train_series.shape[0], train_series.shape[1],  128, 128))
    ds.append(train_series)
train_series = np.array(ds)
train_series = train_series.reshape(train_series.shape[0]*train_series.shape[1], train_series.shape[2], train_series.shape[3], train_series.shape[4])


train_features = train_series[:, :10, :, :]
train_target = train_series[:, 10:, :, :]
train_features = torch.Tensor(train_features)
train_target = torch.Tensor(train_target)
train_loader = DataLoader(TensorDataset(train_features, train_target), batch_size=batch_size)


ds = []
for m in moves:
    test_series = prapare_dataset(m, np.arange(17, 25), 40)[:, 20:, :, :]
    test_series = resize(test_series, (test_series.shape[0], test_series.shape[1],  128, 128))
    ds.append(test_series)
test_series = np.array(ds)
test_series = test_series.reshape(test_series.shape[0]*test_series.shape[1], test_series.shape[2], test_series.shape[3], test_series.shape[4])

test_features = test_series[:, :10, :, :]
test_target = test_series[:, 10:, :, :]
test_features = torch.Tensor(test_features)
test_target = torch.Tensor(test_target)
test_loader = DataLoader(TensorDataset(test_features, test_target), batch_size=batch_size)


device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=[128, 128],
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())
print(model)
model = model.to(device)


losses = []
val_losses = []

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for train_features, train_targets in train_loader:
        train_features = train_features.to(device)
        train_targets = train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(train_features)
        train_loss = criterion(outputs, train_targets)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(train_loader)

    val_loss_value = 0
    for v_train_features, v_train_targets in test_loader:
        v_train_features = v_train_features.to(device)
        v_train_targets = v_train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(v_train_features)
        val_loss = criterion(outputs, v_train_targets)
        val_loss.backward()
        optimizer.step()
        val_loss_value += val_loss.item()
    val_loss_value = val_loss_value / len(test_loader)

    print(f'epoch {epoch}/{epochs}, loss={np.round(loss, 5)}, test_loss={np.round(val_loss_value, 5)}')

    losses.append(loss)
    val_losses.append(val_loss_value)

    if epoch%1000==0:
        torch.save(model.state_dict(), f'models_KTH/KTH_{epoch}_mse.pt')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(np.arange(epochs), losses)
ax1.set_ylabel('MSE Loss')
ax1.set_xlabel('Epoch')
ax1.set_title(f'Train, loss={np.round(losses[-1], 5)}')
ax1.set_yscale('log')
ax2.grid()

ax2.plot(np.arange(epochs), val_losses)
ax2.set_ylabel('MSE Loss')
ax2.set_xlabel('Epoch')
ax2.set_title(f'Validation, loss={np.round(val_losses[-1], 5)}')
ax2.grid()
ax2.set_yscale('log')
plt.suptitle('Convergence plot')
plt.tight_layout()
plt.savefig(f'models_KTH/10to10_convergence_KTH_{epoch}.png')
plt.show()

test_features = test_series[:, :10, :, :]
test_target = test_series[:, 10:, :, :]
print('Data loaded')

for s in range(5):
    tensor_features = torch.Tensor(test_features[s]).to(device)
    prediction = model(tensor_features).detach().cpu().numpy()
    plt.rcParams["figure.figsize"] = (12, 3)
    fig, axs = plt.subplots(2, 10)
    for i in range(10):
        axs[0][i].imshow(prediction[i], cmap='Greys_r')
        axs[0][i].set_title(f't={11+i}')
        axs[1][i].imshow(test_target[s][i], cmap='Greys_r')

        axs[0][i].axes.xaxis.set_ticks([])
        axs[1][i].axes.xaxis.set_ticks([])
        axs[0][i].axes.yaxis.set_ticks([])
        axs[1][i].axes.yaxis.set_ticks([])

    plt.suptitle(f'Sample {s}')
    plt.tight_layout()
    plt.show()

