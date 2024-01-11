import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import time

from tensorflow.keras.preprocessing.image import load_img, img_to_array

imgpath_1 = 'Pos/train/'
imgpath_0 = 'Neg/train/'

image_shape = (21600,224,224,3)

x_traindata = np.empty(image_shape)
i = 0
y_train = []
for image in os.listdir(imgpath_1):
    img = load_img(imgpath_1 + image, target_size = (224, 224))
    x = np.array(img)
    x_traindata[i] = x
    y_train.append(1)
    i += 1
for image in os.listdir(imgpath_0):
    img = load_img(imgpath_0 + image, target_size = (224, 224))
    x = np.array(img)
    x_traindata[i] = x
    y_train.append(0)
    i += 1

x_train = x_traindata / 255

print(x_train)
print(y_train)

x_train = []
y_train = []

testpath_1 = 'Pos/test/'
testpath_0 = 'Neg/test/'

image_shape = (2400,224,224,3)

x_testdata = np.empty(image_shape)
i = 0
y_test = []
for image in os.listdir(testpath_1):
    img = load_img(testpath_1 + image, target_size = (224, 224))
    x = np.array(img)
    x_testdata[i] = x
    y_test.append(1)
    i += 1
for image in os.listdir(testpath_0):
    img = load_img(testpath_0 + image, target_size = (224, 224))
    x = np.array(img)
    x_testdata[i] = x
    y_test.append(0)
    i += 1

x_test = x_testdata / 255

print(x_test)
print(y_test)

x_test = []
y_test = []