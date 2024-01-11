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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess the input image for VGG16
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features from the VGG16 model
def extract_vgg16_features(img_path):
    img_array = preprocess_image(img_path)
    features = base_model.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened

imgpath_1 = 'Pos/1000train/'
imgpath_0 = 'Neg/1000train/'
features = []
y_train = []
for image in os.listdir(imgpath_1):
    features.append(extract_vgg16_features(imgpath_1 + image))
    y_train.append(1)

for image in os.listdir(imgpath_0):
    features.append(extract_vgg16_features(imgpath_0 + image))
    y_train.append(0)


testpath_1 = 'Pos/500test/'
testpath_0 = 'Neg/500test/'
test_features = []
y_test = []
for image in os.listdir(testpath_1):
    test_features.append(extract_vgg16_features(testpath_1 + image))
    y_test.append(1)

for image in os.listdir(testpath_0):
    test_features.append(extract_vgg16_features(testpath_0 + image))
    y_test.append(0)


clf = SVC(kernel='linear')

clf.fit(features, y_train)

y_pred = clf.predict(test_features)

# accuracy: (tp + tn) / (p + n)
accuracy_svm = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy_svm)
# precision tp / (tp + fp)
precision_svm = precision_score(y_test, y_pred)
print('Precision: %f' % precision_svm)
# recall: tp / (tp + fn)
recall_svm = recall_score(y_test, y_pred)
print('Recall: %f' % recall_svm)
# f1: 2 tp / (2 tp + fp + fn)
f1_svm = f1_score(y_test, y_pred)
print('F1 score: %f' % f1_svm)


##1000train500test
#Accuracy: 0.565000
#Precision: 0.538190
#Recall: 0.916000
#F1 score: 0.678016