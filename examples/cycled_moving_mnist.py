import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def get_cycled_data(cycles_num, array):
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    return arr


data = np.load('data/moving_mnist.npy').astype(np.float32) / 255
data = data[:, 3, :, :]

train = get_cycled_data(5, data)
test = get_cycled_data(1, data)


from torchcnnbuilder.preprocess.time_series import multi_output_tensor, single_output_tensor

train_dataset = multi_output_tensor(data=train,
                                    pre_history_len=15,
                                    forecast_len=5)
test_dataset = single_output_tensor(data=test,
                                    forecast_len=5)



from torch import nn, optim, tensor
from torchcnnbuilder.models import ForecasterBase
import time
device = 'cuda'
model = ForecasterBase(input_size=[64, 64],
                       in_time_points=15,
                       out_time_points=5,
                       n_layers=5,
                       finish_activation_function=nn.ReLU(inplace=True))
model = model.to(device)
epochs = 1000
batch_size = 10
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
losses = []
epochs_list = []
start = time.time()
for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = model(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(dataloader)
    print(f'{epoch}/{epochs} - loss={round(loss, 5)}')
    losses.append(loss)
    epochs_list.append(epoch)

plt.plot(epochs_list, losses)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss history')
plt.show()

test_features = test_dataset[0][0]
test_target = test_dataset[0][1]
tensor_features = tensor(test_features).to(device)
prediction = model(tensor_features).detach().cpu().numpy()
plt.rcParams["figure.figsize"] = (15, 6)
fig, axs = plt.subplots(2, 5)
for i in range(5):
    axs[0][i].imshow(prediction[i])
    axs[1][i].imshow(test_target[i])
fig.show()