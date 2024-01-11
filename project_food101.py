import os, time
os.environ['PYTHONHASHSEED']=str(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
seed = 12
import tensorflow as tf
from tensorflow import keras as k
from keras import backend as K
import os, shutil, re, string
import matplotlib.pyplot as plt
# import spacy
import json
from skimage.transform import resize
from skimage import img_as_ubyte
from imageio import imread
import datetime
def set_seed():
    global seed
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

folder_main = 'Food101'
folder_images = os.path.join(folder_main, 'images')
trainjson_fp = folder_main+'/meta/meta/train.json'
testjson_fp = folder_main+'/meta/meta/test.json'
with open(trainjson_fp, 'r') as f:
    trainjson = json.load(f)
with open(testjson_fp, 'r') as f:
    testjson = json.load(f)
labels = list(trainjson.keys())
# labels[:5] #['churros', 'hot_and_sour_soup', 'samosa', 'sashimi', 'pork_chop']
def readImg(img):
    global folder_images
    return imread(os.path.join(folder_images, img+'.jpg'))

# create training data
train_samples_subset_list = []
for x in trainjson.keys():
    train_samples_subset_list.extend(trainjson[x][:50])
set_seed()
train_samples_subset = list(np.random.permutation(train_samples_subset_list))
print("Total number of samples for train",len(train_samples_subset))
print("Some samples are",train_samples_subset[:5])
# remove single channel image
train_sample_one = ['lasagna/3787908', 'bread_pudding/1375816', 'steak/1340977']
train_samples_subset = [x for x in train_samples_subset if x not in train_sample_one]
print("Remaining samples for train are",len(train_samples_subset))

# create testing data
test_samples_subset_list = []
for x in testjson.keys():
    test_samples_subset_list.extend(testjson[x][:20])
set_seed()
test_samples_subset = list(np.random.permutation(test_samples_subset_list))
print("Total number of samples for test",len(test_samples_subset))
print("Some samples are",test_samples_subset[:5])

# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(labels):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)

img_tensor = [128, 128, 3]

def cropResize(image, y, z):
    return img_as_ubyte(resize(image, (y,z)))

def normalizeImage(image):
    # applying normalization
    return image/255.0

def preprocessImage(image, y, z):
    return normalizeImage(cropResize(image, y, z))

# flip
def flipImage(img):
    return np.flip(img)

def getBatchData(t, batch, batch_size, flip):
    global img_tensor, labels
    [h, w, c] = [img_tensor[0], img_tensor[1], img_tensor[2]]
    batch_data = np.zeros((batch_size,h,w,c)) # generating null image representations
    batch_labels = np.zeros((batch_size, len(labels))) # batch_labels is the one hot representation of output
    if flip:
        batch_data_flip = np.zeros((batch_size,h,w,c))
        batch_labels_flip = np.zeros((batch_size, len(labels)))
    # batch_labels = []
    for idx in range(batch_size): # iterating over the batch_size
        imgPath = t[idx + (batch*batch_size)]
        imgLabel = imgPath.strip().split('/')[0]
        image = readImg(imgPath)
        # separate preprocessImage function is defined for cropping, resizing and normalizing images
        batch_data[idx,:,:,0] = preprocessImage(image[:, :, 0], h, w)
        batch_data[idx,:,:,1] = preprocessImage(image[:, :, 1], h, w)
        batch_data[idx,:,:,2] = preprocessImage(image[:, :, 2], h, w)

        batch_labels[idx, label_index[imgLabel]] = 1

        if flip:
            batch_data_flip[idx,:,:,0] = preprocessImage(flipImage(image[:, :, 0]), h, w)
            batch_data_flip[idx,:,:,1] = preprocessImage(flipImage(image[:, :, 1]), h, w)
            batch_data_flip[idx,:,:,2] = preprocessImage(flipImage(image[:, :, 2]), h, w)

            batch_labels_flip[idx, label_index[imgLabel]] = 1
    if flip:
        batch_data = np.concatenate((batch_data, batch_data_flip))
        batch_labels = np.concatenate((batch_labels, batch_labels_flip))

    return batch_data, batch_labels

def generator(folder_list, batch_size, flip=False):
    print('\nLoading from', len(folder_list), 'images; batch size =', batch_size)
    while True:
        num_batches = int(len(folder_list)/batch_size)
        for batch in range(num_batches): # we iterate over the number of batches
#             print("\rReading batch",str(batch+1),"of total",str(num_batches), end='')
            yield getBatchData(folder_list, batch, batch_size, flip)
        
        # checking if any remaining batches are there or not
        if len(folder_list)%batch_size != 0:
            # updated the batch size and yield
            batch_size_rem = len(folder_list)%batch_size
            yield getBatchData(folder_list, batch, batch_size_rem, flip)


# check complete batch shape
sample_generator = generator(train_samples_subset, batch_size=16, flip=True)
sample_batch_data, sample_batch_labels = next(sample_generator)
print("\nSample Train batch data shape", sample_batch_data.shape)
print("Train Batch labels", sample_batch_labels[0])

# validation batch sample
sample_test_generator = generator(test_samples_subset, batch_size=8)
sample_test_batch_data, sample_test_batch_labels = next(sample_test_generator)
print("\nSample Test batch data shape", sample_test_batch_data.shape)
print("Test Batch labels", sample_test_batch_labels[0])

def plotModelHistory(h):
    fig, ax = plt.subplots(1, 2, figsize=(15,4))
    ax[0].plot(h.history['loss'])   
    ax[0].plot(h.history['val_loss'])
    ax[0].legend(['loss','val_loss'])
    ax[0].title.set_text("Train loss vs Validation loss")

    ax[1].plot(h.history['categorical_accuracy'])   
    ax[1].plot(h.history['val_categorical_accuracy'])
    ax[1].legend(['categorical_accuracy','val_categorical_accuracy'])
    ax[1].title.set_text("Train accuracy vs Validation accuracy")

    print("Max. Training Accuracy", max(h.history['categorical_accuracy']))
    print("Max. Validaiton Accuracy", max(h.history['val_categorical_accuracy']))


logPath = './logs/'
if not os.path.exists(logPath):
    os.mkdir(logPath)
# %load_ext tensorboard
# %tensorboard --logdir logPath

# selected set for training and prediction
train_set = train_samples_subset
test_set = test_samples_subset

class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        ACCURACY_THRESHOLD = 0.90
        if(logs.get('val_categorical_accuracy') > ACCURACY_THRESHOLD):
            print("\n\nStopping training as we have reached %2.2f%% accuracy!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

def trainModel(model, epochs, optimizer, vb=1, modelName='model'):
    global train_set, test_set
    logs = logPath + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 128
    bestModelPath = './'+modelName+'_model.hdf5'
    callback = myCallback()
    cbs = [callback,
           k.callbacks.TensorBoard(log_dir=logs, histogram_freq=1),
           k.callbacks.ReduceLROnPlateau(monitor = 'val_categorical_accuracy',patience = 5, verbose = 1),
           k.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy',patience = 5,verbose = 1,restore_best_weights = True),
           k.callbacks.ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True)]

    # setup generators for training
    train_generator = generator(train_set, batch_size, flip=True)
    test_generator = generator(test_set, batch_size, flip=True)
    
    num_train_sequences = len(train_set)
    print('# training sequences =', num_train_sequences)
    num_test_sequences = len(test_set)
    print('# test sequences =', num_test_sequences)

    if (num_train_sequences%batch_size) == 0:
        steps_per_epoch = int(num_train_sequences/batch_size)
    else:
        steps_per_epoch = (num_train_sequences//batch_size) + 1

    if (num_test_sequences%batch_size) == 0:
        validation_steps = int(num_test_sequences/batch_size)
    else:
        validation_steps = (num_test_sequences//batch_size) + 1
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.CategoricalAccuracy(), k.metrics.Precision(), k.metrics.Recall()])
    return model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                               validation_data=test_generator, validation_steps=validation_steps,
                               verbose=vb, batch_size=batch_size, callbacks=cbs)

def trainModelSingle(model, epochs, optimizer, vb=1, modelName='model'):
    global train_set, test_set
#     logs = logPath + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_size = 128
    bestModelPath = './'+modelName+'_model.hdf5'
    callback = myCallback()
    cbs = [callback,
#            k.callbacks.TensorBoard(log_dir=logs, histogram_freq=1),
           k.callbacks.ReduceLROnPlateau(monitor = 'val_categorical_accuracy',patience = 5, verbose = 1),
           k.callbacks.EarlyStopping(monitor = 'val_categorical_accuracy',patience = 5,verbose = 1,restore_best_weights = True),
           k.callbacks.ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True)]

    # setup generators for training
    train_generator = generator(train_set, batch_size, flip=True)
    test_generator = generator(test_set, batch_size, flip=True)

    num_train_sequences = len(train_set)
    num_test_sequences = len(test_set)

    if (num_train_sequences%batch_size) == 0:
        steps_per_epoch = int(num_train_sequences/batch_size)
    else:
        steps_per_epoch = (num_train_sequences//batch_size) + 1

    if (num_test_sequences%batch_size) == 0:
        validation_steps = int(num_test_sequences/batch_size)
    else:
        validation_steps = (num_test_sequences//batch_size) + 1

    return model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                               validation_data=test_generator, validation_steps=validation_steps,
                               verbose=vb, batch_size=batch_size, callbacks=cbs)

# evaluate model with time
def evaluate(model):
    global test_set
    batch_size = 128
    num_train_sequences = len(test_set)
    steps_per_epoch = 0
    if (num_train_sequences%batch_size) == 0:
        steps_per_epoch = int(num_train_sequences/batch_size)
    else:
        steps_per_epoch = (num_train_sequences//batch_size) + 1

    test_generator = generator(test_set, batch_size=batch_size)
    t1 = time.time()
    model = k.models.load_model(model)
    eval_results = model.evaluate_generator(test_generator, steps=steps_per_epoch)
    t2 = time.time()
    print(f'\nAccuracy: {eval_results[1]}, Loss: {eval_results[0]}')
    print(f'Total Prediction Time: {t2-t1}')
    print(f'FPS Prediction Time: {len(test_set)/(t2-t1)}')
    print(f'Prediction Time per Image: {(t2-t1)/len(test_samples_subset)}')

vgg16 = k.applications.VGG16(weights='imagenet', input_shape=img_tensor, include_top=False)
vgg16.trainable = False

model_3 = k.models.Sequential([
                             vgg16,
                             tf.keras.layers.GlobalAveragePooling2D(),
                             k.layers.Dropout(0.2),
                             k.layers.Dense(512, activation='relu'),
                             k.layers.BatchNormalization(),
                             k.layers.Dropout(0.1),
                             k.layers.Dense(256, activation='relu'),
                             k.layers.BatchNormalization(),
                             k.layers.Dropout(0.1),
                             k.layers.Dense(len(index_label), activation='softmax')
])
print(model_3.summary())

trainModel(model_3, 10, 'adam', modelName='vgg16')

evaluate('./vgg16_model.hdf5')