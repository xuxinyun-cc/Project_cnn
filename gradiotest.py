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

import gradio as gr

def classify_image(inp):
  inp = inp.reshape((-1, 224, 224, 3))
  inp = preprocess_input(inp)
  prediction = model.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(9)}

image = gr.inputs.Image(shape=(224, 224), label="風景照")
label = gr.outputs.Label(num_top_classes=9, label="AI辨識結果")

gr.Interface(fn=classify_image, inputs=image, outputs=label,
             title="照片溝突辨識機",
             description="我能辨識9種不同的照片構圖",
             capture_session=True).launch()
