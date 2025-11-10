import os.path

#################################################################
#         Following block install additional packages used in this example                           #
#         If your environment is already set up, install them manually to avoid version conflicts    #
#################################################################

try:
    import numpy as np
except ImportError:
    print(f'numpy not found, installing')
    import pip
    pip.main(["install", "numpy"])
    import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f'matplotlib not found, installing')
    import pip
    pip.main(["install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    print(f'pandas not found, installing')
    import pip
    pip.main(["install", "pandas"])
    import pandas as pd

try:
    from pytorch_msssim import ssim
except ImportError:
    print(f'pytorch_msssim not found, installing')
    import pip
    pip.main(["install", "pytorch_msssim"])
    from pytorch_msssim import ssim

#################################################################


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchcnnbuilder.preprocess import multi_output_tensor
from torchcnnbuilder.models import ForecasterBase
from data_loader import get_timespatial_series

# This script generate 2D CNN with 5 layers and train it with saving weights of model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')
reduce_spatial = True  # Set True for time effectiveness while test script or debug

sea_name = 'kara'
start_date = '19790101'
end_date = '20200101'
sea_data, dates = get_timespatial_series(sea_name, start_date, end_date)
if reduce_spatial:
    sea_data = sea_data[:, ::2, ::2]
sea_data = sea_data[::7]
dates = dates[::7]

pre_history_size = 104
forecast_size = 52

dataset = multi_output_tensor(data=sea_data,
                              forecast_len=forecast_size,
                              pre_history_len=pre_history_size)
dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
print('Loader created')

encoder = ForecasterBase(input_size=(sea_data.shape[1], sea_data.shape[2]),
                         n_layers=5,
                         in_time_points=pre_history_size,
                         out_time_points=forecast_size)
encoder.to(device)
print(encoder)

optimizer = optim.Adam(encoder.parameters(), lr=0.001)
criterion = nn.L1Loss()

losses = []
start = time.time()
epochs = 1000
best_loss = 999
best_model = None
for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    if loss is None:
        break
    if loss < best_loss and loss is not None:
        print('Upd best model')
        best_model = encoder
        best_loss = loss
    losses.append(loss)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

end = time.time() - start
print(f'Runtime seconds: {end}')
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(encoder.state_dict(), f"models/{sea_name}_{pre_history_size}_{forecast_size}_l1({start_date}-{end_date}){epochs}.pt")
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Runtime={end}')
plt.savefig(f"models/{sea_name}_{pre_history_size}_{forecast_size}_l1({start_date}-{end_date}){epochs}.png")
plt.show()
