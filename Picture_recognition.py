'''-----------------------------------------------------------
							\\\||///
						   { -    - }
				 [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author		:		Yopen Hu
foldername  :		MNIST_train_LeNet
filename    :       Picture_recognition.py
Description :		object recognition in CIFAR10 using 
					CNN --- LeNet 
Date  	  By   Version          Change Description
==============================================================
19/04/13  HYP    0.0         	读取并判别输入数据
19/04/14  HYP    1.0         	计算输入数据的概率
--------------------------------------------------------------
									0ooo
--------------------------ooo0-----(    )---------------------
						 (    )     )  /
						  \  (     (__/
						   \__)
-----------------------------------------------------------'''
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from LeNet import LeNet

# 模型读取
myLeNet = torch.load('Best_LeNet_Model.pkl')
if torch.cuda.is_available():
    myLeNet.cuda()
    print("cuda is available, and the calculation will be moved to GPU\n")
else:
    print("cuda is unavailable!")

# 文件读取->灰度->压缩
frame = cv2.imread('8.png')
originImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame', originImg)
inputImage = cv2.resize(originImg, (28, 28))
inputImage = np.ones([28,28]) * 255 - inputImage

# 灰度域拉伸
inputImage = np.floor(255 / (np.max(np.max(inputImage)) - np.min(np.min(inputImage))) 
				* (inputImage - np.min(np.min(inputImage))))
# cv2.imshow('inputImage', inputImage)
# print(inputImage) 

# 与Net输入做匹配
inputImage = np.expand_dims(inputImage, axis=0)	# 维度增加！！！
inputImage = np.expand_dims(inputImage, axis=0)

# 输入必须归一化！和数据集保持一致
tensorImage = torch.FloatTensor(inputImage/255)
varImage = Variable(tensorImage).cuda()
output,_ = myLeNet(varImage)      # LeNet output
_, correct_label = torch.max(output, 1)  # 输出预测概率的label

# GPUtensor转numpy
npOutput = output.cpu().detach().numpy()[0]
npCorrect_label = correct_label.cpu().detach().numpy()[0]

# print(npCorrect_label)
similarity = np.exp(npOutput)/np.sum(np.exp(npOutput))
print('label:', npCorrect_label, 'similarity :', similarity[npCorrect_label]*100,'%')

# cv2.waitKey(0)

