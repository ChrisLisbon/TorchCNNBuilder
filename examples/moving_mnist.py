import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, tensor
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, PolynomialLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from pytorch_msssim import ssim
from torchcnnbuilder.models import ForecasterBase

data = np.load('data/moving_mnist.npy').astype(np.float32)/255

train_set = data[:, :8000, :, :]
test_set = data[:, 2000:, :, :]

train_features = train_set[:15, :, :, :]
train_features = np.swapaxes(train_features, 0, 1)
train_target = train_set[15:, :, :, :]
train_target = np.swapaxes(train_target, 0, 1)

train_dataset = TensorDataset(tensor(train_features), tensor(train_target))

model = ForecasterBase(input_size=[64, 64],
                       in_channels=15,
                       out_channels=5,
                       n_layers=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)

epochs = 1000
batch_size = 100

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
print('loader created')
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.L1Loss()
losses = []
epoches = []
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(epochs):
    loss = 0

    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)

        optimizer.zero_grad()
        outputs = model(train_features)

        train_loss = criterion(outputs, test_features)
        #train_loss = 1 - ssim(outputs, test_features, data_range=1)

        train_loss.backward()
        optimizer.step()

        loss += train_loss.item()

    scheduler.step(loss)
    loss = loss / len(dataloader)

    '''if len(losses) > 10:
        if losses[-5] - loss < 0.0005:
            print('lr = 0.001')
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        else:
            print('lr = 0.0001')
            for g in optimizer.param_groups:
                g['lr'] = 0.0001'''

    losses.append(loss)
    epoches.append(epoch)

    #if epoch % 10 == 0 or epoch - 1 == epochs:
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

torch.save(model.state_dict(), f'mnist_{epochs}.pt')

plt.plot(epoches, losses)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.title('Loss history')
plt.legend()
plt.show()

test_features = test_set[:15, :, :, :]
test_features = np.swapaxes(test_features, 0, 1)
test_target = test_set[15:, :, :, :]
test_target = np.swapaxes(test_target, 0, 1)

for s in range(test_features.shape[0]):
    tensor_features = tensor(test_features[s]).to(device)
    prediction = model(tensor_features).detach().cpu().numpy()
    plt.rcParams["figure.figsize"] = (15, 6)
    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0][i].imshow(prediction[i])
        axs[1][i].imshow(test_target[s][i])
    fig.show()