import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import time

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

## 訓練資料

# imgpath_1 = 'Pos/train/'
imgpath_1 = 'Pos/1000train/'
# imgpath_0 = 'Neg/train/'
imgpath_0 = 'Neg/1000train/'

image_shape = (2000,224,224,3)

x_train = np.empty(image_shape)
i = 0
y_train = []
for image in os.listdir(imgpath_1):
    img = load_img(imgpath_1 + image, target_size = (224, 224))
    x = np.array(img)
    x_train[i] = x
    y_train.append(1)
    i += 1
for image in os.listdir(imgpath_0):
    img = load_img(imgpath_0 + image, target_size = (224, 224))
    x = np.array(img)
    x_train[i] = x
    y_train.append(0)
    i += 1

x_train = x_train / 255
y_train = to_categorical(y_train, 2)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

base_model.trainable = False

model = Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

# features = model.predict(x_train)

# 創建SVM分類器
# svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 訓練SVM分類器
# svm.fit(features, y_train)


set_optimizer = Adam(learning_rate=0.001) 
model.compile(loss='binary_crossentropy', optimizer=set_optimizer, metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=10)

x_train = []
y_train = []

## 測試資料

# testpath_1 = 'Pos/test/'
testpath_1 = 'Pos/500test/'
# testpath_0 = 'Neg/test/'
testpath_0 = 'Neg/500test/'

image_shape = (1000,224,224,3)

x_test = np.empty(image_shape)
i = 0
y_test = []
for image in os.listdir(testpath_1):
    img = load_img(testpath_1 + image, target_size = (224, 224))
    x = np.array(img)
    x_test[i] = x
    y_test.append(1)
    i += 1
for image in os.listdir(testpath_0):
    img = load_img(testpath_0 + image, target_size = (224, 224))
    x = np.array(img)
    x_test[i] = x
    y_test.append(0)
    i += 1

x_test = x_test / 255
y_test = to_categorical(y_test, 2)

# 模型預測
y_pred = model.predict(x_test)
# y_pred = svm.predict(x_test)


eval = model.evaluate(x_test, y_test)
print(eval)

x_test = []
y_test = []


