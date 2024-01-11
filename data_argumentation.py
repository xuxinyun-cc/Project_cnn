import numpy as np
import cv2

## labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

# oldpath = 'KU_PCP_dataset/all_label.txt'
# oldpath = 'KU_PCP_dataset/train_label.txt'
oldpath = 'KU_PCP_dataset/test_label.txt'
# newpath = 'KU_PCP_dataset/all_data_argu_label.txt'
# newpath = 'KU_PCP_dataset/train_label_data_argu.txt'
newpath = 'KU_PCP_dataset/test_label_data_argu.txt'

# read = 'KU_PCP_dataset/all/'
# read = 'KU_PCP_dataset/train_img/'
read = 'KU_PCP_dataset/test_img_data_argu/'
# save = 'KU_PCP_dataset/all_data_argu/'
# save = 'KU_PCP_dataset/train_img_data_argu/'
save = 'KU_PCP_dataset/test_img_data_argu/'

data = np.loadtxt(oldpath, delimiter=' ')

count = 0
RoT = []
vertical = []
horizontal = []
diagonal = []
curved = []
triangle = []
center = []
symmetric = []
pattern = []

for d in data:
    # print(d)
    count += 1
    if sum(d)==1:
        if d[0]==1:
            RoT.append(count)
        if d[1]==1:
            vertical.append(count)
        if d[2]==1:
            horizontal.append(count)
        if d[3]==1:
            diagonal.append(count)
        if d[4]==1:
            curved.append(count)
        if d[5]==1:
            triangle.append(count)
        if d[6]==1:
            center.append(count)
        if d[7]==1:
            symmetric.append(count)
        if d[8]==1:
            pattern.append(count)


print("RoT_len:",len(RoT))
print("vertical_len:",len(vertical))
print("horizontal_len:",len(horizontal))
print("diagonal_len:",len(diagonal))
print("curved_len:",len(curved))
print("triangle_len:",len(triangle))
print("center_len:",len(center))
print("symmetric_len:",len(symmetric))
print("pattern_len:",len(pattern))

# new = 4245
# new = 7479
# new = 8923
# new = 9478

# new = 3170
# new = 5796
# new = 7092
# new = 7750
# new = 7961

# new = 1076
# new = 2200
# new = 2684
new = 2867

# #RoT
# total_RoT = 0
# for num in RoT:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     # output_0 = cv2.flip(img,0) #上下翻轉
#     # output_1 = cv2.flip(img,1) #左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_0)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_1)
#     # new += 1

#     w, h = img.shape[0], img.shape[1]
#     output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     cv2.imwrite(save + str(new) + ".jpg", output_s)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_l)
#     new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_RoT += 2
#     if total_RoT >= 83:
#         break
# if total_RoT < 83:
#     print("RoT不夠，只有",total_RoT)

# #vertical
# total_vertical = 0
# for num in vertical:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     # output_0 = cv2.flip(img,0) #上下翻轉
#     # output_1 = cv2.flip(img,1) #左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_0)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_1)
#     # new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     # output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     # cv2.imwrite(save + str(new) + ".jpg", output_s)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_l)
#     # new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_cutr = img[0:w, 0:h-100]
#     # output_cutl = img[0:w, 100:h]
#     # cv2.imwrite(save + str(new) + ".jpg", output_cutr)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_cutl)
#     # new += 1

#     output_2 = cv2.flip(img,-1) #上下左右翻轉
#     cv2.imwrite(save + str(new) + ".jpg", output_2)
#     new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_ss = cv2.resize(img, (int(h*0.25), int(w*0.25)))
#     # output_ll = cv2.resize(img, (int(h*1.75), int(w*1.75)))
#     # cv2.imwrite(save + str(new) + ".jpg", output_ss)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_ll)
#     # new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     # f.write(line)

#     total_vertical += 1
#     if total_vertical >= 2:
#         break
# if total_vertical < 2:
#     print("vertical不夠，只有",total_vertical)

# #horizontal
# total_horizontal = 0
# for num in horizontal:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     output_0 = cv2.flip(img,0) #上下翻轉
#     output_1 = cv2.flip(img,1) #左右翻轉
#     cv2.imwrite(save + str(new) + ".jpg", output_0)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_1)
#     new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_horizontal += 2
#     if total_horizontal >= 139:
#         break
# if total_horizontal < 139:
#     print("horizontal不夠，只有",total_horizontal)

# #diagonal
# total_diagonal = 0
# for num in diagonal:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     output_0 = cv2.flip(img,0) #上下翻轉
#     output_1 = cv2.flip(img,1) #左右翻轉
#     cv2.imwrite(save + str(new) + ".jpg", output_0)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_1)
#     new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_diagonal += 2
#     if total_diagonal >= 161:
#         break
# if total_diagonal < 161:
#     print("diagonal不夠，只有",total_diagonal)

# #curved
# total_curved = 0
# for num in curved:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     # output_0 = cv2.flip(img,0) #上下翻轉
#     # output_1 = cv2.flip(img,1) #左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_0)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_1)
#     # new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     # output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     # cv2.imwrite(save + str(new) + ".jpg", output_s)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_l)
#     # new += 1

#     # output_2 = cv2.flip(img,-1) #上下左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_2)
#     # new += 1

#     w, h = img.shape[0], img.shape[1]
#     output_ss = cv2.resize(img, (int(h*0.25), int(w*0.25)))
#     output_ll = cv2.resize(img, (int(h*1.75), int(w*1.75)))
#     cv2.imwrite(save + str(new) + ".jpg", output_ss)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_ll)
#     new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_curved += 2
#     if total_curved >= 33:
#         break
# if total_curved < 33:
#     print("curved不夠，只有",total_curved)

# #triangle
# total_triangle = 0
# for num in triangle:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     output_0 = cv2.flip(img,0) #上下翻轉
#     output_1 = cv2.flip(img,1) #左右翻轉
#     cv2.imwrite(save + str(new) + ".jpg", output_0)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_1)
#     new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     # output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     # cv2.imwrite(save + str(new) + ".jpg", output_s)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_l)
#     # new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_triangle += 2
#     if total_triangle >= 178:
#         break
# if total_triangle < 178:
#     print("triangle不夠，只有",total_triangle)

# #center
# total_center = 0
# for num in center:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     output_0 = cv2.flip(img,0) #上下翻轉
#     output_1 = cv2.flip(img,1) #左右翻轉
#     cv2.imwrite(save + str(new) + ".jpg", output_0)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_1)
#     new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_center += 2
#     if total_center >= 39:
#         break
# if total_center < 39:
#     print("center不夠，只有",total_center)

# #symmetric
# total_symmetric = 0
# for num in symmetric:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     # output_0 = cv2.flip(img,0) #上下翻轉
#     # output_1 = cv2.flip(img,1) #左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_0)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_1)
#     # new += 1

#     w, h = img.shape[0], img.shape[1]
#     output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     cv2.imwrite(save + str(new) + ".jpg", output_s)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_l)
#     new += 1

#     # output_2 = cv2.flip(img,-1) #上下左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_2)
#     # new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_symmetric += 2
#     if total_symmetric >= 94:
#         break
# if total_symmetric < 94:
#     print("symmetric不夠，只有",total_symmetric)

# #pattern
# total_pattern = 0
# for num in pattern:
#     if num < 10:
#         name = "000" + str(num)
#     elif num>=10 and num<100:
#         name = "00" + str(num)
#     elif num>=100 and num<1000:
#         name = "0" + str(num)
#     elif num>=1000:
#         name = str(num)
    
#     img = cv2.imread(read + name + ".jpg")
#     # output_0 = cv2.flip(img,0) #上下翻轉
#     # output_1 = cv2.flip(img,1) #左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_0)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_1)
#     # new += 1

#     # w, h = img.shape[0], img.shape[1]
#     # output_s = cv2.resize(img, (int(h*0.5), int(w*0.5)))
#     # output_l = cv2.resize(img, (int(h*1.5), int(w*1.5)))
#     # cv2.imwrite(save + str(new) + ".jpg", output_s)
#     # new += 1
#     # cv2.imwrite(save + str(new) + ".jpg", output_l)
#     # new += 1

#     w, h = img.shape[0], img.shape[1]
#     output_cutr = img[0:w, 0:h-100]
#     output_cutl = img[0:w, 100:h]
#     cv2.imwrite(save + str(new) + ".jpg", output_cutr)
#     new += 1
#     cv2.imwrite(save + str(new) + ".jpg", output_cutl)
#     new += 1

#     # output_2 = cv2.flip(img,-1) #上下左右翻轉
#     # cv2.imwrite(save + str(new) + ".jpg", output_2)
#     # new += 1

#     with open(oldpath,'r') as op:
#         for row,line in enumerate(op):
#             if row == (num-1):
#                 with open(newpath, 'a') as f:
#                     f.write(line)
#                     f.write(line)

#     total_pattern += 2
#     if total_pattern >= 43:
#         break
# if total_pattern < 43:
#     print("pattern不夠，只有",total_pattern)