
# Semantic Segmentation in Keras-Tensorflow using U-Net #


## Author ##

Dennis Núñez Fernández  
[https://dennishnf.com](https://dennishnf.com) 


## Description ##

This repository provides basic example of semantic segmentation using the U-Net architecture, and also using the frameworks Keras and Tensorflow. You can run the example and train your own model using your own dataset. The example is intended to be executed in two independent ways: in Google Colab and in Native Python.

The repository makes was written in python 3.6. It's expected to run in any version of Tensorflow 1.x without problems, so, it was tested in the versions of Keras and Tensorflow as follows: 

- Version: Google Colav (main.ipynb): Tensorflow 1.15.0, Keras 2.1.5
- Version: Native Python (main.py): Tensorflow 1.10.0, Keras 2.1.5


## Version: Google Colab (main.ipynb) ##

You should use the file main.ipynb.

Note 1: I recommend DOWNLOADING the folder and then UPLOADING the folder to your Google Drive, and modify some paths in the main.ipynb file of some labs according to your path in order to work properly.

Note 2: For datasets, zip files and temporal location in the tmp folder at Google Colab space were used because extracting data from this is faster compared to extracting data from your Google drive.


## Version: Native Python (main.py) ##

You should use the file main.py.

I recommend to install Anaconda. Then create an environment, and work in that enviroment.

Install the required packages:

```
pip install -r requirements.txt
```

Finally, open your IDE (like Spyder, PyCharm, etc) and run the example.

Note: Modify some paths in the main.ipynb file of some labs according to your path in order to work properly.
