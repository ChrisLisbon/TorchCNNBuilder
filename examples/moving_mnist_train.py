import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, tensor
from torch.utils.data import TensorDataset, DataLoader
from torchcnnbuilder.models import ForecasterBase

from skimage.transform import resize

data = np.load('data/moving_mnist.npy').astype(np.float32)/255

train_set = data[:, :1000, :, :]
validation_set = data[:, 1000:1500:, :, :]
test_set = data[:, 1500:2000, :, :]

dim = 45

train_set = resize(train_set, (train_set.shape[0], train_set.shape[1], dim, dim))
validation_set = resize(validation_set, (validation_set.shape[0], validation_set.shape[1], dim, dim))
test_set = resize(test_set, (test_set.shape[0], test_set.shape[1], dim, dim))

train_features = train_set[:10, :, :, :]
train_features = np.swapaxes(train_features, 0, 1)
train_target = train_set[10:, :, :, :]
train_target = np.swapaxes(train_target, 0, 1)
train_dataset = TensorDataset(tensor(train_features), tensor(train_target))

validation_features = validation_set[:10, :, :, :]
validation_features = np.swapaxes(validation_features, 0, 1)
validation_target = validation_set[10:, :, :, :]
validation_target = np.swapaxes(validation_target, 0, 1)
validation_dataset = TensorDataset(tensor(validation_features), tensor(validation_target))

device = 'cuda'
print(f'Calculation on device: {device}')
model = ForecasterBase(input_size=[dim, dim],
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=5,
                       finish_activation_function=nn.ReLU())

optimizer = optim.AdamW(model.parameters(), lr=0.001)

'''failed_t = 10000
file_w = f'models_mnist/45_mnist_{failed_t}(small).pt'
file_o = f'models_mnist/optim_45_mnist_{failed_t}(small).pt'

model.load_state_dict(torch.load(file_w))
model.train()

optimizer.load_state_dict(torch.load(file_w))

'''
model = model.to(device)

epochs = np.arange(1, 1000000)
batch_size = 500
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()


losses = []
val_losses = []
epoches = []
for epoch in epochs:
    loss = 0
    for train_features, train_targets in dataloader:
        train_features = train_features.to(device)
        train_targets = train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(train_features)
        train_loss = criterion(outputs, train_targets)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(dataloader)

    val_loss_value = 0
    for v_train_features, v_train_targets in val_dataloader:
        v_train_features = v_train_features.to(device)
        v_train_targets = v_train_targets.to(device)
        optimizer.zero_grad()
        outputs = model(v_train_features)
        val_loss = criterion(outputs, v_train_targets)
        val_loss.backward()
        optimizer.step()
        val_loss_value += val_loss.item()
    val_loss_value = val_loss_value / len(val_dataloader)

    print(f'epoch {epoch}, loss={np.round(loss, 5)}, validation_loss={np.round(val_loss_value, 5)}')

    losses.append(loss)
    val_losses.append(val_loss_value)
    epoches.append(epoch)
    if epoch % 5000 == 0:
        torch.save(model.state_dict(), f'models_mnist/{dim}_mnist_{epoch}(small).pt')
        torch.save(optimizer.state_dict(), f'models_mnist/optim_{dim}_mnist_{epoch}(small).pt')
        df = pd.DataFrame()
        df['epoch'] = epoches
        df['train_loss'] = losses
        df['val_losses'] = val_losses
        df.to_csv(f'models_mnist/{dim}_mnist_{epoch}(small).csv', index=False)

torch.save(model.state_dict(), f'models_mnist/{dim}_mnist_{epoch}(small).pt')