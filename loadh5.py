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

# testpath = 'KU_PCP_dataset/test_img/'
# testpath = 'KU_PCP_dataset/test_img_data_argu/'
# testpath = 'Pos/train/'
testpath = 'Neg/train/'

# test_shape = (100,224,224,3)
# test_shape = (1084,224,224,3)
test_shape = (10912,224,224,3)

x_test = np.empty(test_shape)
i = 0

for test in os.listdir(testpath):
    testimg = load_img(testpath + test, target_size = (224, 224))
    x = np.array(testimg)
    x_test[i] = x
    i += 1

# testpath = 'Pos/test/'
testpath = 'Neg/test/'

for test in os.listdir(testpath):
    testimg = load_img(testpath + test, target_size = (224, 224))
    x = np.array(testimg)
    x_test[i] = x
    i += 1

x_test = x_test / 255


# 模型預測
predictions = model.predict(x_test)
# print(predictions)
test_pred = np.argmax(predictions, axis=1)
# print(test_pred)
predicted_val = []
for i in test_pred:
    predicted_val.append(labels[i])
# print(predicted_val)

# 將預測轉換為二進制矩陣
# y_pred_binary = (y_pred > 0.5).astype(int)
# y_true_binary = y_test

# 計算 precision、recall 和 F1 分數
# precision_weighted = precision_score(y_true_binary, y_pred_binary, average='weighted')
# precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro')
# precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro')
# recall_weighted = recall_score(y_true_binary, y_pred_binary, average='weighted')
# recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro')
# recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro')
# f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted')
# f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro')
# f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro')

# print(f'Precision_weighted: {precision_weighted:.4f}')
# print(f'Precision_macro: {precision_macro:.4f}')
# print(f'Precision_micro: {precision_micro:.4f}')
# print(f'Recall_weighted: {recall_weighted:.4f}')
# print(f'Recall_macro: {recall_macro:.4f}')
# print(f'Recall_micro: {recall_micro:.4f}')
# print(f'F1 Score_weighted: {f1_weighted:.4f}')
# print(f'F1 Score_macro: {f1_macro:.4f}')
# print(f'F1 Score_micro: {f1_micro:.4f}')

# eval = model.evaluate(x_test, y_test)
# print(eval)
    
name = [i+1 for i in range(10912)]
df = pd.DataFrame({'id':name, 'predict_label':predicted_val})
print(df)
df.to_csv("projectGPD_neg_all.csv", index=False)