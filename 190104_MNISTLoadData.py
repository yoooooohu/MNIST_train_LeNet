'''-----------------------------------------------------------
                            \\\||///
                           { -    - }
	             [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author      :		Yopen Hu
foldername  :		MNIST_train_LeNet
filename    :       MNISTLoadData.py
Description :		object segmentation in CIFAR10 using 
				    CNN --- LeNet 
Date  	  By   Version          Change Description
==============================================================
19/01/04  HYP    0.0       	 extract data & read data
--------------------------------------------------------------
                                    0ooo
--------------------------ooo0-----(    )---------------------
                         (    )     )  /
                          \  (     (__/
                           \__)
-----------------------------------------------------------'''
# library
# standard library
import os 
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data 	     # dataset
import torchvision	                     # torch vision library
import matplotlib.pyplot as plt 		 # plot 2D data, Visualization
 
torch.manual_seed(1)    # reproducible 设定生成随机数的种子，并返回一个 torch._C.Generator 对象

DOWNLOAD_MNIST = False

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
 
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST
)
 
# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[100].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[100])
plt.show()
