import os
import pandas as pd
import numpy as np

labels = ['RoT', 'vertical', 'horizontal', 'diagonal', 'curved', 'triangle', 'center', 'symmetric', 'pattern']

tpath = 'KU_PCP_dataset/test_label.txt'
y_test = np.loadtxt(tpath, delimiter=' ')

testpath = 'KU_PCP_dataset/test_img/'
name = []
for test in os.listdir(testpath):
    name.append(test.split('.')[0])

ans1 = []
ans2 = []
ans3 = []
count = 0
num,classnum = y_test.shape
for i in range(num):
    k = 0
    count = int(sum(y_test[i]))
    temp = 0     
    
    if count==3:
        for j in range(classnum):
            if y_test[i][j] == 1:
                temp += 1
                if temp==1: 
                    ans1.append(labels[j])
                elif temp==2:
                    ans2.append(labels[j])
                elif temp==3:
                    ans3.append(labels[j])
    elif count==2:
        for j in range(classnum):
            if y_test[i][j] == 1:
                temp += 1
                if temp==1: 
                    ans1.append(labels[j])
                elif temp==2:
                    ans2.append(labels[j])
        ans3.append('None')
    elif count==1:
        for j in range(classnum):
            if y_test[i][j] == 1:
                temp += 1
                ans1.append(labels[j])
        ans2.append('None')
        ans3.append('None')
    elif count==0:
        ans1.append('None')
        ans2.append('None')
        ans3.append('None')
                
df = pd.DataFrame({'id': name,'ans1': ans1,'ans2': ans2,'ans3': ans3})
print(df)
df.to_csv("1084_test_labels.csv", index=False)
# print(count)