import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD

labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

## x_train

# imgpath = '100_train_img/'
imgpath = 'KU_PCP_dataset/train_img/'

# image_shape = (100,224,224,3)
image_shape = (3169,224,224,3)
x_train = np.empty(image_shape)
i = 0
for image in os.listdir(imgpath):
    img = load_img(imgpath + image, target_size = (224, 224))
    x = np.array(img)
    x_train[i] = x
    i += 1

x_train = x_train / 255

## y_train (答案)

# filepath = '100_train_label.txt'
filepath = 'KU_PCP_dataset/train_label.txt'
y_train = np.loadtxt(filepath, delimiter=' ')

## x_test

# testpath = '100_test_image/'
testpath = 'KU_PCP_dataset/test_img/'

# test_shape = (100,224,224,3)
test_shape = (1084,224,224,3)
x_test = np.empty(test_shape)
i = 0
name = []
for test in os.listdir(testpath):
    name.append(test.split('.')[0])
    testimg = load_img(testpath + test, target_size = (224, 224))
    x = np.array(testimg)
    x_test[i] = x
    i += 1

x_test = x_test / 255

## y_test

# tpath = '100_test_label.txt'
tpath = 'KU_PCP_dataset/test_label.txt'
y_test = np.loadtxt(tpath, delimiter=' ')

ans = []
num,classnum = y_test.shape
for i in range(num):
    k = 0
    for j in range(classnum):
        if y_test[i][j] == 1:
            ans.append(labels[j])
            # print("{} : {}".format(i,labels[j]))
            # print(len(ans))
            break
        else:
            k += 1
    if k == 9:
        ans.append('None')

## Load a convolutional base with pre-trained weights

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

base_model.trainable = False

model = Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dense(9, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=10)


predictions = model.predict(x_test)
# print(predictions)
test_pred = np.argmax(predictions, axis=1)
# print(test_pred)
predicted_val = []
for i in test_pred:
    predicted_val.append(labels[i])
# print(predicted_val)
eval = model.evaluate(x_test, y_test)
print("loss:", eval[0])
print('accuracy:', eval[1])