[[Click For Russian version]](README_RU.md)

## Sea Ice Concentration Forecasting Example

The ice condition data was prepared based on the OSI SAF product.  You can download a prepared sample 
dataset to run the example [here](https://disk.yandex.ru/d/C8KrnPCr65nqSw).

After downloading the data, you need to specify the data directory in the [```data_loader```](data_loader.py) file - ```/path_to_data/```.

**Note:** The downloaded file `kara.rar` is an archive. You need to extract it before using.

**Extraction commands:**

*   **For Linux/macOS:**
    ```bash
    unrar x kara.rar
    ```
    *(If `unrar` is not installed, you can usually get it via `sudo apt-get install unrar` on Ubuntu/Debian)*

*   **For Windows:**
    You can use the built-in "Extract to..." command by right-clicking on the file in Explorer, or use a tool like [7-Zip](https://www.7-zip.org/). In the Command Prompt:
    ```cmd
    "C:\Program Files\7-Zip\7z.exe" x kara.rar
    ```
    *(Adjust the path if 7-Zip is installed in a different location)*

The [```data_loader```](data_loader.py) file contains a function for loading specific time intervals for ice concentration matrices.

After downloading the data and configuring the directory, you can proceed to train  a convolutional neural network model by running the [```train_cnn```](train_cnn.py) script.