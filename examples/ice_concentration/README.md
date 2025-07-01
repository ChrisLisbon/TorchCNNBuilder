## Sea Ice Concentration Forecasting Example

The ice condition data was prepared based on the OSI SAF product.  
You can download a prepared sample dataset to run the example [here](https://disk.yandex.ru/d/C8KknPCr65nqSw).

After downloading the data, you need to specify the data directory in the  
[```data_loader```](data_loader.py) file - ```/path_to_data/```.

The [```data_loader```](data_loader.py) file contains a function for loading specific  
time intervals for ice concentration matrices.

After downloading the data and configuring the directory, you can proceed to train  
a convolutional neural network model by running the [```train_cnn```](train_cnn.py) script.