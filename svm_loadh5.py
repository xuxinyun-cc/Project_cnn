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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model = load_model("best_model_f1_m(data_argu_new)_ori.h5", custom_objects={"precision_m":precision_m, "recall_m":recall_m, "f1_m":f1_m})

testpath = 'Food101/images/hamburger/'

test_shape = (1000,224,224,3)

x_test = np.empty(test_shape)
i = 0

for test in os.listdir(testpath):
    testimg = load_img(testpath + test, target_size = (224, 224))
    x = np.array(testimg)
    x_test[i] = x
    i += 1

x_test = x_test / 255


predictions = model.predict(x_test)
# print(predictions)
test_pred = np.argmax(predictions, axis=1)
# print(test_pred)
predicted_val = []
for i in test_pred:
    predicted_val.append(labels[i])


model_svm = load_model("vgg16_svm_allgpd_allfood_1_model.hdf5")

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_vgg16_features(img_path):
    img_array = preprocess_image(img_path)
    features = model_svm.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened


imgpath_1 = 'Pos/train/'
imgpath_0 = 'Neg/train/'

x_train = []
y_train = []
for image in os.listdir(imgpath_1):
    x_train.append(extract_vgg16_features(imgpath_1 + image))
    y_train.append(1)

for image in os.listdir(imgpath_0):
    x_train.append(extract_vgg16_features(imgpath_0 + image))
    y_train.append(0)


testpath = 'Food101/images/hamburger/'


x_test_svm = []
for image in os.listdir(testpath):
    x_test_svm.append(extract_vgg16_features(testpath + image))


svm_model = svm.SVC()

svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test_svm)


name = [i+1 for i in range(1000)]
df = pd.DataFrame({'id':name, 'predict_svm':y_pred, 'predict_label':predicted_val})
print(df)
df.to_csv("project_hambuger.csv", index=False)