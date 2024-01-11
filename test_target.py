import os
import numpy as np
import pandas as pd
## y_train (答案)

labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

trainpath = 'KU_PCP_dataset/train_label.txt'
train = np.loadtxt(trainpath, delimiter=' ')
train_ans = []
num,classnum = train.shape
# for i in range(num):
#     k = 0
#     for j in range(classnum):
#         if train[i][j] == 1:
#             train_ans.append(labels[j])
#             break
#         else:
#             k += 1
#     if k == 9:
#         train_ans.append('None')
# print(train_ans)
print(train)

# filepath = 'KU_PCP_dataset/test_label.txt'
# test = np.loadtxt(filepath, delimiter=' ')
# test_ans = []
# num,classnum = test.shape
# for i in range(num):
#     k = 0
#     for j in range(classnum):
#         if test[i][j] == 1:
#             test_ans.append(labels[j])
#             break
#         else:
#             k += 1
#     if k == 9:
#         test_ans.append('None')
# # # print(test_ans)

# df = pd.DataFrame({'train_label':test_ans})
# # print(df)
# df.to_csv("data_test.csv", index=False)


# filepath = 'KU_PCP_dataset/test_label.txt'
# filepath = '100_test_label.txt'
# fix = []
# target = []
# k = 0
# with open(filepath) as f:
#     lines = f.readlines()
#     for line in lines:
#         s = line.strip().split(' ')
#         print(s)
            
# print(type(s))

# print(target)
# fix = np.array(target)
# print(fix)




# filepath = '100_test_label.txt'
# ## 讀取標籤檔案並建立訓練標籤陣列
# train_labels = np.loadtxt(filepath, delimiter=' ')
# ans = []
# x,y = train_labels.shape
# for i in range(x):
#     k = 0
#     for j in range(y):
#         if train_labels[i][j] == 1:
#             ans.append(labels[j])
#             print("{} : {}".format(i,labels[j]))
#             print(len(ans))
#             break
#         else:
#             k += 1
#     if k == 9:
#         ans.append('None')
# print(len(ans))
# print(ans)


# 確認訓練標籤陣列的形狀
# print("Train labels shape:", train_labels.shape)
# print(train_labels)

## x_train

# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet_v2 import preprocess_input


# imgpath = '100_test_image/'
# imgpath = 'KU_PCP_dataset/test_img/'

# image_shape = (100,1024,720,3)
# img_data = np.empty(image_shape)
# i = 0
# img = []
# for image in os.listdir(imgpath):
#     img.append(image.split('.')[0])
    # img = load_img(imgpath + image)
    # img = img.resize((720, 1024))
    # x = np.array(img)
    # img_data[i] = x
    # i += 1
# print(img)
# print(len(img))

# img_data = img_data / 255
# print(img_data)
# img_data = np.array(img_data)
# img_data = preprocess_input(img_data)

# imgpath = '100_train_img/'
# imgpath = 'KU_PCP_dataset/train_img/'
# imgpath = '10_test_img/'

# image_shape = (100,224,224,3)
# image_shape = (3169,224,224,3)
# x_train = np.empty(image_shape)
# i = 0
# name = []
# for image in os.listdir(imgpath):
    #print(image)
    # print(image.split('.')[0])
    # img = load_img(imgpath + image, target_size = (224, 224))
    # # img = img.resize((720, 1024))
    # x = np.array(img)
    # x_train[i] = x
    # i += 1

# x_train = x_train / 255
# print(x_train)
