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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.animation")

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

from tools import synthetic_time_series, save_gif, create_logger, log_print
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

seed = 42
np.random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_print(logger, f'seed sets {seed}')
log_print(logger, f'device sets {device}')

# ------------------------------------
# iterations setup
# ------------------------------------
input_shape = (55, 55)
pre_history_len = 120
forecast_len = 30
batch_size = 50
epochs = 2_000

"""
    Each case can be run with future variables:

     iterations_path = 'iterations - batchnorm L1 loss'
     iterations_path = 'iterations - batchnorm SSIM loss'
     iterations_path = 'iterations - batchnorm BCE loss'
    
     iterations_path = 'iterations - none L1 loss'
     iterations_path = 'iterations - none SSIM loss'
     iterations_path = 'iterations - none BCE loss'
"""

iterations_path = 'iterations - batchnorm SSIM + BCE loss'  # name of experiment - depends on normalization and loss function
if not os.path.exists(iterations_path):
    os.makedirs(iterations_path, exist_ok=True)

"""
    Each type of normalization can be tested with keys:         
        'none': None
"""
model_type = {
    'batchnorm': 'batchnorm',
    #'none': None
}
"""
    Each type of loss can be tested with keys:
         'L1 loss': nn.L1Loss(),
         'SSIM loss': lambda x, y: 1 - ssim(x, y, data_range=1, size_average=True),
         'BCE loss': lambda x, y: nn.functional.binary_cross_entropy(torch.clamp(x, 0, 1), y),
"""
loss_type = {
    'SSIM + BCE loss': lambda x, y: 0.6 * nn.functional.binary_cross_entropy(torch.clamp(x, 0, 1), y) + 0.4 * (
                1 - ssim(torch.clamp(x, 0, 1), y, data_range=1, size_average=True)),
}

noise_type = {'0%': 0,
              '1%': 0.01,
              '3%': 0.03,
              '5%': 0.05,
              '10%': 0.1,
              '25%': 0.25,
              '50%': 0.5}

results = {'norm type': [],
           'loss type': [],
           'noise type': [],
           'loss value': [],
           'L1 val': [],
           'SSIM val': [],
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
exp_number = 0
for k1, v1 in noise_type.items():

    if v1 == 0:
        noised_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        noised_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # percentage mask for noise
        mask = np.full(input_shape, False)
        num_true = int(np.prod(input_shape) * v1)
        indices = np.random.choice(np.prod(input_shape), num_true, replace=False)
        mask.flat[indices] = True

        # creating noise for train/test
        noised_train = np.random.normal(loc=0, scale=1, size=train.shape) * mask
        noised_test = np.random.normal(loc=0, scale=1, size=test.shape) * mask

        # clipping values
        noised_train = np.clip(train + noised_train, 0, 1)
        noised_test = np.clip(test + noised_test, 0, 1)

        # saving noise gif
        save_gif(noised_train, f'{iterations_path}/noise_example{k1}')

        # creating noised datasets
        noised_train_dataset = multi_output_tensor(data=noised_train,
                                                   pre_history_len=pre_history_len,
                                                   forecast_len=forecast_len)
        noised_test_dataset = single_output_tensor(data=noised_test,
                                                   forecast_len=forecast_len)

        # creating noised dataloaders
        noised_train_dataloader = DataLoader(noised_train_dataset, batch_size=batch_size, shuffle=False)
        noised_test_dataloader = DataLoader(noised_test_dataset, batch_size=batch_size, shuffle=False)

    for k2, v2 in model_type.items():
        for k3, v3 in loss_type.items():

            # logging
            exp_path = f'{iterations_path}/{exp_number}_{k3.split()[0]}_{k1}_{k2}'
            log_print(logger, f'start {exp_path}')
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
            exp_logger = create_logger(f'{exp_path}/exp.log')

            # recording the results
            results['noise type'].append(k1)
            results['norm type'].append(k2)
            results['loss type'].append(k3)
            loss_history = []

            # for each model there is its own scheduler and loss
            optim_params, scheduler_params = dict(), dict()
            finish_activation = nn.ReLU(inplace=True)

            if k2 == 'batchnorm':
                if k3 == 'L1 loss':
                    optim_params = {'lr': 1e-3}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.9,
                                        'patience': 20,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-7}

                elif k3 == 'SSIM loss':
                    optim_params = {'lr': 1e-3}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.9,
                                        'patience': 40,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-6}
                    finish_activation = nn.Sigmoid()

                elif k3 == 'BCE loss':
                    optim_params = {'lr': 1e-3}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.9,
                                        'patience': 20,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-7}
                    finish_activation = nn.Sigmoid()

                elif k3 == 'SSIM + BCE loss':
                    optim_params = {'lr': 1e-3}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.9,
                                        'patience': 40,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-7}
                    finish_activation = nn.Sigmoid()
                else:
                    print('No opt and scheduler')

            elif k2 == 'none':
                if k3 == 'L1 loss':
                    optim_params = {'lr': 1e-4}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.95,
                                        'patience': 15,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-8}
                    epochs = 1_000
                elif k3 == 'SSIM loss':
                    optim_params = {'lr': 1e-4}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.95,
                                        'patience': 30,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-7}
                    epochs = 1000
                elif k3 == 'BCE loss':
                    optim_params = {'lr': 1e-3}
                    scheduler_params = {'mode': 'min',
                                        'factor': 0.9,
                                        'patience': 20,
                                        'threshold': 1e-5,
                                        'threshold_mode': 'abs',
                                        'min_lr': 1e-7}
                    finish_activation = nn.Sigmoid()
                    epochs = 1000
                else:
                    print('No opt and scheduler')

            # training
            criterion = v3
            model = ForecasterBase(input_size=input_shape,
                                   in_time_points=pre_history_len,
                                   out_time_points=forecast_len,
                                   n_layers=5,
                                   normalization=v2,
                                   finish_activation_function=finish_activation).to(device)

            optimizer = optim.AdamW(model.parameters(), **optim_params)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

            loss = 0
            start_time = datetime.datetime.now()

            model.train()
            for epoch in range(epochs):
                for noised_batch, batch in zip(noised_train_dataloader, train_dataloader):
                    X = noised_batch[0].to(device)
                    Y = batch[1].to(device)

                    optimizer.zero_grad()
                    outputs = model(X)

                    train_loss = criterion(outputs, Y)

                    train_loss.backward()
                    optimizer.step()
                    loss += train_loss.item()

                loss = loss / len(train_dataloader)
                loss_history.append(loss)
                scheduler.step(loss)
                log_print(exp_logger,
                          f"-- epoch : {epoch + 1}/{epochs}, recon loss = {loss}, lr = {scheduler.get_last_lr()[-1]}")

                if epoch in (0, int(epochs // 2 * 0.8), epochs // 2, int(epochs * 0.75)):
                    save_gif(outputs[0].detach().cpu().numpy(), f'{exp_path}/predict_epoch_{epoch}')
                    save_gif(Y[0].detach().cpu().numpy(), f'{exp_path}/y_epoch_{epoch}')

            end_time = datetime.datetime.now()

            # recording the results
            results['loss value'].append(loss)
            results['start time'].append(start_time)
            results['end time'].append(end_time)
            time_difference = end_time - start_time
            results['duration (sec)'].append(time_difference.total_seconds())

            # validation
            model.eval()
            with torch.no_grad():
                for noised_batch, batch in zip(noised_test_dataloader, test_dataloader):
                    X = noised_batch[0].to(device)
                    Y = batch[1].to(device)

                    outputs = model(X)

                    l1_val = nn.L1Loss()(outputs, Y).item()
                    ssim_val = ssim(outputs, Y, data_range=1, size_average=False)[0].item()

            # recording the results
            results['L1 val'].append(l1_val)
            results['SSIM val'].append(ssim_val)

            # saving results
            df_results = pd.DataFrame(results)
            df_results.to_csv('results.csv', index=False, sep='\t')

            exp_number += 1

            # saving loss history
            plt.figure()
            plt.plot(list(range(len(loss_history))), loss_history)
            plt.grid()
            plt.xlabel('Epoch')
            plt.ylabel(f'{k3}')
            plt.title('Loss history')
            plt.savefig(f'{exp_path}/loss_history.png')
            plt.close()

            # saving model
            torch.save(model.state_dict(), f'{exp_path}/model.pth')

            # saving gif
            save_gif(outputs[0].detach().cpu().numpy(), f'{exp_path}/predict')
            save_gif(X[0].detach().cpu().numpy(), f'{exp_path}/x')
            save_gif(Y[0].detach().cpu().numpy(), f'{exp_path}/y')

            log_print(logger, f'end {exp_path}')

log_print(logger, 'end')

df_results = pd.DataFrame(results)
df_results.to_csv(f'{iterations_path}/results.csv', index=False, sep='\t')
