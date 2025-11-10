[[Click For Russian version]](README_RU.md)

## Sea Ice Concentration Forecasting Example

The ice condition data was prepared based on the OSI SAF product.  You can download a prepared sample 
dataset to run the example [here](https://disk.yandex.ru/d/sdj5K7wh-unXdA).

After downloading the data, you need to specify the data directory in the [```data_loader```](data_loader.py) file - ```/path_to_data/```.

**Note:** The downloaded file `kara.zip` is an archive. You need to extract it before using.

**Extraction commands:**

*   **For Linux/macOS:**
    ```bash
    unzip kara.zip
    ```
    *(If `unzip` is not installed, use your package manager:*
    - **Ubuntu/Debian:** `sudo apt-get install unzip`
    - **CentOS/RHEL/Fedora:** `sudo yum install unzip` or `sudo dnf install unzip`
    - **Arch Linux/Manjaro:** `sudo pacman -S unzip`
    - **macOS:** `brew install unzip`*)*

*   **For Windows:**
    You can use the built-in "Extract All..." command by right-clicking on the file in Explorer. No additional software needed!
    
    *Alternatively, in Command Prompt:*
    ```cmd
    tar -xf kara.zip
    ```
    *(Built-in in Windows 10 and later)*

The [```data_loader```](data_loader.py) file contains a function for loading specific time intervals for ice concentration matrices.

After downloading the data and configuring the directory, you can proceed to train  a convolutional neural network model by running the [```train_cnn```](train_cnn.py) script.