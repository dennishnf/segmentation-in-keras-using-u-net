# -*- coding: utf-8 -*-

"""

Semantic Segmentation using the U-Net

Caution: modify the path /content/drive/My Drive/segmentation-in-keras-using-u-net in the 
lines of code of this file, according to your path in Drive. Eg: if the location of your folder 
is in the main path of your Google Drive under the name u-net-colab, then modify the last path to: 
/content/drive/My Drive/u-net-colab

"""

#%%

# Downloading the dataset

#%%

# Download manually and unzip the daatset from:
# https://docs.google.com/uc?export=download&id=1mjPuhgbjcadbyCV0CRp-C4w4YtuvuhRo

#%%

# Preparing Libraries

#%%

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow
import keras
tensorflow.test.gpu_device_name()


#%%

print(tensorflow.__version__)
print(keras.__version__)

# should print something like:
# 1.10.0
# 2.1.5

#%%

folder_location = "/home/dennis/Downloads/segmentation-in-keras-using-u-net/"
dataset_location = "/home/dennis/Downloads/segmentation-in-keras-using-u-net/dataset-membranes/"

#%%

# Import Modules

#%%

import sys
import os

sys.path.append(os.path.abspath(folder_location))

import model
import data

from model import *
from data import *

#%%

# Preprocess the data

#%%

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,dataset_location+'train','image','label',data_gen_args,save_to_dir = None)



#%%

# Define model

#%%

model = unet()
model.summary()

#%%

# Training

#%%

model_checkpoint = ModelCheckpoint(folder_location+'unet_weights.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=200,epochs=3,callbacks=[model_checkpoint])

#%%

# Testing

#%%

# Note: Given an image, generates the result in a single image where a certain pixel 
# corresponds to the probability of matching the class. So, we use a thresholding to 
# convert from a image composed with probabilities to a binary image

testGene = testGenerator(dataset_location+"test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult(dataset_location+"test",results,thresholding=0.5)

#%%

# Testing Visualization

#%%

from PIL import Image
import matplotlib.pyplot as plt
import numpy

#%%

plt.figure(figsize=(6,10))

for i in range(3*2):
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img1=Image.open(dataset_location+"test/"+str(i)+".png")
    img2=Image.open(dataset_location+"test/"+str(i)+"_predict.png")
    if i%2==0:
      plt.imshow(img1,cmap='gray')
    if i%2==1:
      plt.imshow(img2,cmap='gray')

plt.show()

#%%

plt.figure(figsize=(10,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    img1=Image.open(dataset_location+"test/"+str(i)+".png")
    img1=img1.resize((200, 200), Image.ANTIALIAS)
    img2=Image.open(dataset_location+"test/"+str(i)+"_predict.png")
    img2=img2.resize((200, 200), Image.ANTIALIAS)
    
    rgbimg1=Image.new("RGB",img1.size)
    rgbimg1.paste(img1)
    rgbimg2=Image.new("RGB",img2.size)
    rgbimg2.paste(img2)
    
    rgbimg1=numpy.asarray(rgbimg1)
    rgbimg2=numpy.asarray(rgbimg2)
    
    rgbimg2_ = rgbimg2.copy()
    rgbimg2_[rgbimg2_ >= 125] = 255
    rgbimg2_[rgbimg2_ < 125] = 0
    rgbimg2__ = rgbimg2_.copy()
   
    for i in range(0,200):
      for j in range(0,200):
        rgbimg2__[i][j] = [rgbimg2_[i][j][0],rgbimg2_[i][j][1],0]
    
    plt.imshow(rgbimg1, cmap='gray') # I would add interpolation='none'
    plt.imshow(rgbimg2__, cmap='gray', alpha=0.3) # interpolation='none'
    plt.axis('off')

plt.show()

#%%

# Inference

#%%

# Note: Given an image, generates the result in a single image where a certain pixel 
# corresponds to the probability of matching the class. So, we use a thresholding to 
# convert from a image composed with probabilities to a binary image

from model import *
from data import *

testGene = testGenerator(dataset_location+"test")
model = unet()
model.load_weights(folder_location+"unet_weights.hdf5")
results = model.predict_generator(testGene,15,verbose=1)
saveResult(dataset_location+"test",results,thresholding=0.5)

#%%

# Inference Visualization

#%%

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%

img1=mpimg.imread(dataset_location+"test/0.png")
plt.imshow(img1, cmap = 'gray')
plt.axis('off')
plt.show()

#%%

img2=mpimg.imread(dataset_location+"test/0_predict.png")
plt.imshow(img2, cmap = 'gray')
plt.axis('off')
plt.show()

#%%