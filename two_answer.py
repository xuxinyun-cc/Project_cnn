import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Input
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



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 凍結基本模型的權重
for layer in base_model.layers:
    layer.trainable = False

inputs = Input(shape=(224, 224, 3))

x = base_model(inputs)

x = Flatten()(x)

outputs_x = Dense(9, activation='softmax')(x)

y = base_model(inputs)

y = Flatten()(y)

outputs_y = Dense(9, activation='sigmoid')(y)

model = Model(inputs=inputs, outputs=[outputs_x, outputs_y])

model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train,[y_train, y_train], validation_split=0.2, batch_size=64, epochs=10)


predictions_x,predictions_y = model.predict(x_test)
# print(predictions)
test_pred_x = np.argmax(predictions_x, axis=1)
test_pred_y = np.argmax(predictions_y, axis=1)
# print(test_pred)
predicted_val_x = []
predicted_val_y = []
for i in test_pred_x:
    predicted_val_x.append(labels[i])
for i in test_pred_y:
    predicted_val_y.append(labels[i])
# print(predicted_val)
# eval_x,eval_y = model.evaluate(x_test, y_test)
# print("loss_x:", eval_x[0])
# print('accuracy_x:', eval_x[1])
# print("loss_y:", eval_y[0])
# print('accuracy_y:', eval_y[1])
df = pd.DataFrame({'id':name, 'predict_label_x':predicted_val_x, 'predict_label_y':predicted_val_y, 'true_label':ans})
# print(df)
df.to_csv("1084_test_twoanswer.csv", index=False)