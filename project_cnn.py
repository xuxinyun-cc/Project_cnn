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

labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

## x_train

# imgpath = 'KU_PCP_dataset/train_img/'
# imgpath = 'KU_PCP_dataset/all/'
imgpath = 'KU_PCP_dataset/train_img_data_argu/'

# image_shape = (3169,224,224,3)
# image_shape = (4244,224,224,3)
image_shape = (8080,224,224,3)  ##用這個就不用weight-loss

x_data = np.empty(image_shape)
i = 0
for image in os.listdir(imgpath):
    img = load_img(imgpath + image, target_size = (224, 224))
    x = np.array(img)
    x_data[i] = x
    i += 1
x_train = x_data / 255
# print(x_data)

# filepath = 'KU_PCP_dataset/train_label.txt'
# filepath = 'KU_PCP_dataset/all_label.txt'
filepath = 'KU_PCP_dataset/train_label_data_argu.txt'

y_train = np.loadtxt(filepath, delimiter=' ')

# X is your feature data, y is your class labels
# skf = KFold(n_splits=5, shuffle=True, random_state=42)

# for train_index, test_index in skf.split(x_data, y_data):
#     x_train, x_test = x_data[train_index], x_data[test_index]
#     y_train, y_test = y_data[train_index], y_data[test_index]
    # Train and evaluate your model on X_train, y_train, X_test, y_test


# def create_model(dropout_rate=0.3):

# Load a convolutional base with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

base_model.trainable = False

model = Sequential()

model.add(base_model)
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(3, 3)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='softmax'))
# model.add(Dense(9, kernel_regularizer=l2(0.001),activation='softmax'))

feature_ex = model.predict(x_train)
features = feature_ex.reshape(feature_ex.shape[0],-1)

# 創建SVM分類器
svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 訓練SVM分類器
svm.fit(features, y_train) 



# 有9個類別，training是您的訓練標籤
# num_classes = 9
# training = np.random.randint(num_classes, size=(3396,))

# total_samples = len(y_train)
# class_counts = np.sum(y_train, axis=0)

# class_weight = {}
# for i in range(len(class_counts)):
    # class_weight[i] = total_samples / len(class_counts) * class_counts[i]

# class_weight_dict = class_weight

# 計算每個類別的權重
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(training), y=training)

# 將權重映射為字典
# class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

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


# Compile & train
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
set_optimizer = Adam(learning_rate=0.001) 
# model.compile(loss='categorical_crossentropy', optimizer=set_optimizer, metrics=['accuracy', Precision(), Recall(), tfa.metrics.F1Score(num_classes=9, average='weighted', threshold=0.5)])
# model.compile(loss='categorical_crossentropy', optimizer=set_optimizer, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=set_optimizer, metrics=['accuracy',precision_m, recall_m, f1_m])

model.summary()

# return model

# 設定 ModelCheckpoint 回調
# loss_checkpoint = ModelCheckpoint("best_model_loss.h5", 
#                                   monitor='val_loss',  # 監控驗證集的損失
#                                   save_best_only=True, 
#                                   mode='min',  # 以最小值為目標
#                                   verbose=1)
# accuracy_checkpoint = ModelCheckpoint("best_model_accuracy.h5", 
#                                       monitor='val_accuracy',  # 監控驗證集的準確率
#                                       save_best_only=True, 
#                                       mode='max',  # 以最大值為目標
#                                       verbose=1)
# f1_score_checkpoint = ModelCheckpoint("best_model_f1_m(data_argu_new)_1layer64333233D64.h5", 
#                                       monitor='val_f1_m',  # 監控驗證集的準確率
#                                       save_best_only=True, 
#                                       mode='max',  # 以最大值為目標
#                                       verbose=1)


# 將Keras模型包裝成Scikit-Learn分類器
# model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64)

# 定義參數網格
# param_grid = {
#     'dropout_rate': [0.2, 0.3, 0.4, 0.5]
# }

# 使用GridSearchCV進行網格搜索
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
# grid_result = grid.fit(x_train, y_train)

# 輸出最佳參數和分數
# print("Best parameters found: ", grid_result.best_params_)
# print("Best accuracy found: ", grid_result.best_score_)


# model.fit(x_train, y_train, batch_size=10, epochs=10)
# history = model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=1)
# 在 model.fit 中加入weight loss
# history = model.fit(x_train, y_train, validation_split=0.2, batch_size=256, epochs=10, class_weight=class_weight_dict)
# history = model.fit(x_train, y_train, validation_split=0.2, batch_size=256, epochs=10, class_weight=class_weight_dict, callbacks=[f1_score_checkpoint])
# 在 model.fit 中使用回調
# history = model.fit(x_train, y_train, validation_split=0.2, batch_size=256, epochs=10, callbacks=[loss_checkpoint, accuracy_checkpoint])
# history = model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=1, callbacks=[f1_score_checkpoint])


# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# train_precision = history.history['precision']
# val_precision = history.history['val_precision']
# train_recall = history.history['recall']
# val_recall = history.history['val_recall']
# train_f1_score = history.history['f1_score']
# val_f1_score = history.history['val_f1_score']

# plt.figure()
# # plt.subplot(2, 2, 1)
# plt.plot(train_loss, label='train_loss')
# plt.plot(val_loss, label='val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss_value')
# plt.legend()
# plt.title('loss')

# plt.subplot(2, 2, 2)
# plt.plot(train_precision, label='train_precision')
# plt.plot(val_precision, label='val_precision')
# plt.xlabel('epochs')
# plt.ylabel('precision_value')
# plt.legend()
# plt.title('precision')

# plt.subplot(2, 2, 3)
# plt.plot(train_recall, label='train_recall')
# plt.plot(val_recall, label='val_recall')
# plt.xlabel('epochs')
# plt.ylabel('recall_value')
# plt.legend()
# plt.title('recall')

# plt.subplot(2, 2, 4)
# plt.plot(train_f1_score, label='train_f1_score')
# plt.plot(val_f1_score, label='val_f1_score')
# plt.xlabel('epochs')
# plt.ylabel('f1_score_value')
# plt.legend()
# plt.title('f1_score')

# plt.tight_layout()
# plt.show()

## x_test

# testpath = 'KU_PCP_dataset/test_img/'
testpath = 'KU_PCP_dataset/test_img_data_argu/'

# test_shape = (100,224,224,3)
# test_shape = (1084,224,224,3)
test_shape = (2902,224,224,3)
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

# tpath = 'KU_PCP_dataset/test_label.txt'
tpath = 'KU_PCP_dataset/test_label_data_argu.txt'
y_test = np.loadtxt(tpath, delimiter=' ')

# ans = []
# num,classnum = y_test.shape
# for i in range(num):
#     k = 0
#     for j in range(classnum):
#         if y_test[i][j] == 1:
#             ans.append(labels[j])
#             # print("{} : {}".format(i,labels[j]))
#             # print(len(ans))
#             break
#         else:
#             k += 1
#     if k == 9:
#         ans.append('None')

# 模型預測
y_pred_ex = model.predict(x_test)
y_pred_features = y_pred_ex.reshape(y_pred_ex.shape[0],-1)
y_pred = svm.predict(y_pred_features)
# print(predictions)
# test_pred = np.argmax(predictions, axis=1)
# print(test_pred)
# predicted_val = []
# for i in test_pred:
#     predicted_val.append(labels[i])
# print(predicted_val)

# 將預測轉換為二進制矩陣
y_pred_binary = (y_pred > 0.5).astype(int)
y_true_binary = y_test

# 計算混淆矩陣
# conf_matrix = confusion_matrix(y_true_binary, y_pred_binary)

# 計算 precision、recall 和 F1 分數
precision_weighted = precision_score(y_true_binary, y_pred_binary, average='weighted')
precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro')
precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro')
recall_weighted = recall_score(y_true_binary, y_pred_binary, average='weighted')
recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro')
recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro')
f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted')
f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro')
f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro')

print(f'Precision_weighted: {precision_weighted:.4f}')
print(f'Precision_macro: {precision_macro:.4f}')
print(f'Precision_micro: {precision_micro:.4f}')
print(f'Recall_weighted: {recall_weighted:.4f}')
print(f'Recall_macro: {recall_macro:.4f}')
print(f'Recall_micro: {recall_micro:.4f}')
print(f'F1 Score_weighted: {f1_weighted:.4f}')
print(f'F1 Score_macro: {f1_macro:.4f}')
print(f'F1 Score_micro: {f1_micro:.4f}')

# print(conf_matrix)

eval = model.evaluate(x_test, y_test)
print(eval)
# print("loss:", eval[0])
# print('accuracy:', eval[1])
# df = pd.DataFrame({'id':name, 'predict_label':predicted_val, 'true_label':ans})
# print(df)
# df.to_csv("excel/project_batch256_dropout0.3_savemodel.csv", index=False)