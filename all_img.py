##移動圖片
# import os
# path = 'KU_PCP_dataset/test_img/'
# destination = 'KU_PCP_dataset/try/'
# files = os.listdir(path)
# # print(files)

# index = 3170
# for i in files: #因為資料夾裡面的檔案都要重新更換名稱
#     oldname = path + i #指出檔案現在的路徑名稱，[n]表示第n個檔案
#     newname = destination + str(index) + '.jpg'
#     os.rename(oldname,newname)
#     print(oldname + '>>>' + newname)
#     index = index + 1

# import os
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # imgpath = 'KU_PCP_dataset/all/'
# imgpath = 'KU_PCP_dataset/all_data_argu/'
# # image_shape = (4244,224,224,3)
# image_shape = (9512,224,224,3)
# x_data = np.empty(image_shape)
# i = 0
# for image in os.listdir(imgpath):
#     img = load_img(imgpath + image, target_size = (224, 224))
#     x = np.array(img)
#     x_data[i] = x
#     i += 1

# x_data = x_data / 255

# # filepath = 'KU_PCP_dataset/all_label.txt'
# filepath = 'KU_PCP_dataset/all_data_argu_label.txt'
# y_data = np.loadtxt(filepath, delimiter=' ')


# ##重新分割
# # from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import KFold

# # X is your feature data, y is your class labels
# skf = KFold(n_splits=5, shuffle=True, random_state=42)

# for train_index, test_index in skf.split(x_data, y_data):
#     x_train, x_test = x_data[train_index], x_data[test_index]
#     y_train, y_test = y_data[train_index], y_data[test_index]
#     # Train and evaluate your model on X_train, y_train, X_test, y_test

# print('len(x_train)', len(x_train))  #3396   #7610
# print('len(x_test)', len(x_test))   #848   #1902
# print('len(y_train)', len(y_train))   #3396   #7610
# print('len(y_test)', len(y_test))   #848   #1902


##計算權重
# total_samples = len(y_data)
# print(total_samples)
# class_counts = np.sum(y_data, axis=0)
# print(len(class_counts))
# print(class_counts)

# class_weight = {}
# for i in range(len(class_counts)):

#     class_weight[i] = total_samples / len(class_counts) * class_counts[i]

# print(class_weight)


##翻轉照片
# import cv2
# img = cv2.imread("KU_PCP_dataset/0230_triangle.jpg")
# output_0 = cv2.flip(img,0)
# output_1 = cv2.flip(img,1)
# cv2.imwrite('test_0.jpg',output_0)
# cv2.imwrite('test_1.jpg',output_1)

##複製標籤
# oldpath = 'KU_PCP_dataset/all_label.txt'

# newpath = 'KU_PCP_dataset/all_data_argu_label.txt'

# with open(oldpath,'r') as r:
#     for num,line in enumerate(r):
#         if num == 20: #(第21行)
#             print(line)
#             with open(newpath, 'a') as f:
#                 f.write(line)
#                 f.write(line)

##放大縮小照片
# import cv2
# img = cv2.imread("KU_PCP_dataset/0371-symmetric.jpg")
# img = cv2.imread("KU_PCP_dataset/0230_triangle.jpg")
# w, h = img.shape[0], img.shape[1]
# print(img.shape)
# size = w+200
# scale = h/size
# w_size = int(w/scale)
# output = cv2.resize(img, (int(h*0.5), int(w*0.5)))
# cv2.imwrite('test_2.jpg',output)
# print(output.shape)

##裁切照片
# import cv2
# img = cv2.imread("KU_PCP_dataset/0371-symmetric.jpg")
# # img = cv2.imread("KU_PCP_dataset/0230_triangle.jpg")
# w, h = img.shape[0], img.shape[1]
# print(img.shape)
# output = img[0:w, 0:h-100]
# output = img[0:w, 100:h]
# cv2.imwrite('test_3.jpg',output)
# print(output.shape)


##test rename
# import os
# import cv2

# readpath = 'KU_PCP_dataset/test_img/'
# savepath = 'KU_PCP_dataset/test_img_data_argu/'
# oldname = os.listdir(readpath)
# num = 1
# for old in oldname:
#     img = cv2.imread(readpath + old)
#     if num < 10:
#         newname = "000" + str(num)
#     elif num>=10 and num<100:
#         newname = "00" + str(num)
#     elif num>=100 and num<1000:
#         newname = "0" + str(num)
#     elif num>=1000:
#         newname = str(num)
#     cv2.imwrite(savepath + newname + ".jpg", img)
#     num += 1
# print(oldname)