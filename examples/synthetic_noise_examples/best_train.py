
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
    print(f'matplotlib not found, installing')
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

from tools import synthetic_time_series, save_gif, create_logger, log_print, IoU
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from torchcnnbuilder.models import ForecasterBase
from torchcnnbuilder.preprocess import multi_output_tensor, single_output_tensor

import datetime
import logging

# ------------------------------------
# setting up an experiment
# ------------------------------------
with open('experiment.log', 'w') as f:
    f.write('')
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    filename='experiment.log')
log_print(logger, 'start')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_print(logger, f'device sets {device}')

# ------------------------------------
# iterations setup
# ------------------------------------
input_shape = (55, 55)
pre_history_len = 120
forecast_len = 30
batch_size = 50
epochs = 3_000

results = {'norm type': [None],
           'loss type': ['BCE loss'],
           'noise type': ['0%'],
           'loss value': [],
           'BCE val': [],
           'L1 val': [],
           'SSIM val': [],
           'IoU val': [],
           'start time': [],
           'end time': [],
           'duration (sec)': []}

# ------------------------------------
# data preparation
# ------------------------------------
_, data = synthetic_time_series(num_frames=360,
                                matrix_size=input_shape[0],
                                square_size=10)
train = np.array(data + data + data)
test = np.array(data[10:160])

log_print(logger, f'Train dataset len: {len(train)}, One matrix shape: {train[0].shape}')
log_print(logger, f'Test dataset len: {len(test)}, One matrix shape: {test[0].shape}')

# creating datasets
train_dataset = multi_output_tensor(data=train,
                                    pre_history_len=pre_history_len,
                                    forecast_len=forecast_len)
test_dataset = single_output_tensor(data=test,
                                    forecast_len=forecast_len)

# checking train_dataset
for batch in train_dataset:
    log_print(logger, f'X train shape: {batch[0].shape}')
    log_print(logger, f'Y train shape: {batch[1].shape}')
    break
log_print(logger, f'Dataset len (number of batches): {len(train_dataset)}')

# checking test_dataset
for batch in test_dataset:
    log_print(logger, f'X test shape: {batch[0].shape}')
    log_print(logger, f'Y test shape: {batch[1].shape}')
log_print(logger, f'Dataset len (number of batches): {len(test_dataset)}')

# creating dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------
# iterations
# ------------------------------------
# logging
exp_path = 'best model/'
log_print(logger, f'start {exp_path}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
exp_logger = create_logger(f'{exp_path}exp.log')

# recording the results
loss_history = []
bce_val_history = []
l1_val_history = []
ssim_val_history = []
iou_val_history = []

# training
model = ForecasterBase(input_size=input_shape,
                       in_time_points=pre_history_len,
                       out_time_points=forecast_len,
                       n_layers=5,
                       normalization=None,
                       finish_activation_function=nn.Sigmoid()).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.9,
                                                 patience=20,
                                                 threshold=1e-5,
                                                 threshold_mode='abs',
                                                 min_lr=1e-7)
criterion = nn.BCELoss()

loss = 0
start_time = datetime.datetime.now()

model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        X = batch[0].to(device)
        Y = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(X)

        train_loss = criterion(outputs, Y)

        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    # saving loss
    loss = loss / len(train_dataloader)
    loss_history.append(loss)
    scheduler.step(loss)
    log_print(exp_logger,
              f"-- epoch : {epoch + 1}/{epochs}, recon loss = {loss}, lr = {scheduler.get_last_lr()[-1]}")

    if epoch in (0, int(epochs // 2 * 0.8), epochs // 2, int(epochs * 0.75)):
        save_gif(outputs[0].detach().cpu().numpy(), f'{exp_path}predict_epoch={epoch}')
        save_gif(Y[0].detach().cpu().numpy(), f'{exp_path}y_epoch={epoch}')

    # validation
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            X = batch[0].to(device)
            Y = batch[1].to(device)

            outputs = model(X)

            bce_val = nn.BCELoss()(outputs, Y).item()
            l1_val = nn.L1Loss()(outputs, Y).item()
            ssim_val = ssim(outputs, Y, data_range=1, size_average=False)[0].item()
            iou_val = IoU(outputs, Y, threshold=0.6).item()

    bce_val_history.append(bce_val / len(test_dataloader))
    l1_val_history.append(l1_val / len(test_dataloader))
    ssim_val_history.append(ssim_val / len(test_dataloader))
    iou_val_history.append(iou_val / len(test_dataloader))

end_time = datetime.datetime.now()

# recording the results
results['loss value'].append(loss)
results['start time'].append(start_time)
results['end time'].append(end_time)
time_difference = end_time - start_time
results['duration (sec)'].append(time_difference.total_seconds())

# recording the results
results['BCE val'].append(bce_val)
results['L1 val'].append(l1_val)
results['SSIM val'].append(ssim_val)
results['IoU val'].append(iou_val)

# saving results
df_results = pd.DataFrame(results)
df_results.to_csv(f'{exp_path}results.csv', index=False, sep='\t')

# saving loss history
plt.figure()
plt.plot(list(range(len(loss_history))), loss_history, label='train')
plt.plot(list(range(len(bce_val_history))), bce_val_history, label='val')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(f'BCE')
plt.title('BCE history')
plt.legend()
plt.savefig(f'{exp_path}train_history.png')

# saving ssim history
plt.figure()
plt.plot(list(range(len(ssim_val_history))), ssim_val_history)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(f'SSIM value')
plt.title('SSIM validation history')
plt.legend()
plt.savefig(f'{exp_path}ssim_history.png')

# saving l1 history
plt.figure()
plt.plot(list(range(len(l1_val_history))), l1_val_history, label='loss')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(f'L1 value')
plt.title('L1 validation history')
plt.legend()
plt.savefig(f'{exp_path}l1_history.png')

# saving iou history
plt.figure()
plt.plot(list(range(len(iou_val_history))), iou_val_history, label='loss')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(f'IoU')
plt.title('IoU validation history')
plt.legend()
plt.savefig(f'{exp_path}iou_history.png')

# saving model
torch.save(model.state_dict(), f'{exp_path}model.pth')

# saving gif
save_gif(outputs[0].detach().cpu().numpy(), f'{exp_path}predict')
save_gif(X[0].detach().cpu().numpy(), f'{exp_path}x')
save_gif(Y[0].detach().cpu().numpy(), f'{exp_path}y')

log_print(logger, 'end')
