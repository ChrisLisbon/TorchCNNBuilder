from datetime import datetime
from torch import nn
from torchcnnbuilder.models import ForecasterBase

# Code calculate and print time of model's building for different size of input data
# Код рассчитывает и выводит в консоль время сборки модели для разных размеров входных данных

start = datetime.now()
model = ForecasterBase(input_size=(3840, 2160),
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=50,
                       activation_function=nn.ReLU())
end = datetime.now()
print(f'Size (3840, 2160) 4k: Build in {end-start}')

start = datetime.now()
model = ForecasterBase(input_size=(500, 500),
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=50,
                       activation_function=nn.ReLU())
end = datetime.now()
print(f'Size (500, 500): Build in {end-start}')

start = datetime.now()
model = ForecasterBase(input_size=(150, 150),
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=50,
                       activation_function=nn.ReLU())
end = datetime.now()
print(f'Size (150, 150): Build in {end-start}')


# Code show compatibility of built model with CUDA
# Код выводит совместимость собранной модели с CUDA

model = ForecasterBase(input_size=(125, 125),
                       in_time_points=10,
                       out_time_points=10,
                       n_layers=5,
                       activation_function=nn.ReLU())
actual_device = next(model.parameters()).device
print(f'Device of built model: {actual_device}')
model.to('cuda')
actual_device = next(model.parameters()).device
print(f'Device of built model: {actual_device}')
