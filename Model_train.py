'''-----------------------------------------------------------
                            \\\||///
                           { -    - }
	             [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author		:		Yopen Hu (CSDN ID:Yooo Hu)
foldername  :		MNIST_train_LeNet 
filename    :       Model_train.py
Description :		object segmentation in CIFAR10 using 
					CNN --- LeNet 
Date  	  By     Version          Change Description
==============================================================
19/01/04  Yopen    0.0         extract data & read data
19/01/04  Yopen    1.1       LeNet construction & utilize
19/01/06  Yopen    2.0         Figrue Time Decumentation
19/01/06  Yopen    2.1               Add Test Part
19/04/10  Yopen    3.0               	 Debug
--------------------------------------------------------------
                                    0ooo
--------------------------ooo0-----(    )---------------------
                         (    )     )  /
                          \  (     (__/
                           \__)
-----------------------------------------------------------'''
# standard library
import os 
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data 		# dataset
import torchvision						# torch vision library
import matplotlib.pyplot as plt 		# plot 2D data, Visualization
import visdom 							# python -m visdom.server
from LeNet import LeNet

torch.manual_seed(1)    # reproducible 设定生成随机数的种子，并返回一个 torch._C.Generator 对象

# Hyper Parameters 超参数
EPOCH = 250             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8192		# Data number per batch
LR = 0.001              # learning rate
VISDOM = False           # 绘图

if not os.path.exists('./ModelBackup/'):
    os.makedirs('./ModelBackup/')

##############################################################################################
########################### Mnist digits dataset preparation #################################
DOWNLOAD_MNIST = False  # 是否下载
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

MNISTtrainData = torchvision.datasets.MNIST(            # data size (60000, 28, 28) + label(60000)
    root = './mnist/',
    train = True,                                     # this is training data
    transform = torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download = DOWNLOAD_MNIST
)
# Data Loader for easy mini-batch return in training,
# the image batch shape will be (BATCH_SIZE, 1, 28, 28)
train_loader = Data.DataLoader(dataset=MNISTtrainData, batch_size=BATCH_SIZE, shuffle=True)

MNISTtestData = torchvision.datasets.MNIST(            # data size (10000, 28, 28) + label(10000)
    root = './mnist/', train = False,
    transform = torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download = DOWNLOAD_MNIST
)
test_loader = Data.DataLoader(dataset=MNISTtestData, batch_size=BATCH_SIZE, shuffle=True)

##############################################################################################
######################## define LeNet construction & Parameters ##############################

myLeNet = LeNet()	# 网络实例化

if torch.cuda.is_available():
    myLeNet.cuda()
    print("cuda is available, and the calculation will be moved to GPU\n")
else:
    print("cuda is unavailable!")

##############################################################################################
###########################       training & testing         #################################
optimizer = torch.optim.Adam(myLeNet.parameters(), lr = LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()           # 交叉熵 loss the target label is not one-hotted

if VISDOM:
    vis = visdom.Visdom(env=u'MNIST for leNet')
    vis.line(X=torch.Tensor([1]),Y=torch.Tensor([0]),win='trainAcc', \
        opts=dict(title = 'acc rate(%) for train data', ytickmin = 0, ytickmax = 100)) 
    vis.line(X=torch.Tensor([1]),Y=torch.Tensor([0]),win='trainLoss', \
        opts=dict(title = 'train loss', ytickmin = 0, ytickmax = 2.5))
    vis.line(X=torch.Tensor([1]),Y=torch.Tensor([0]),win='testAcc', \
        opts=dict(title = 'acc rate(%) for test data', ytickmin = 0, ytickmax = 100)) 
    vis.line(X=torch.Tensor([1]),Y=torch.Tensor([0]),win='testLoss', \
        opts=dict(title = 'test loss', ytickmin = 0, ytickmax = 2.5))

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   
        print(x.shape) 
        # gives batch data, normalize x when iterate train_loader
        if torch.cuda.is_available():
            batch_x = Variable(x).cuda()
            batch_y = Variable(y).cuda()
        else:
            batch_x = Variable(x)
            batch_y = Variable(y)

        output, last_layer = myLeNet(batch_x)      # LeNet output
        loss = loss_func(output, batch_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
 
        _, correct_label = torch.max(output, 1)  # 输出预测概率的label
        # print('label', correct_label)
        correct_num = (correct_label == batch_y).sum()

        trainAcc = correct_num.item() / float(batch_y.size(0))

        print('train: Epoch [%d/%d], Iter [%2d/%d] Loss: %.4f, Acc: %.4f' % \
              	(epoch + 1, EPOCH, step + 1, len(MNISTtrainData) // BATCH_SIZE + 1, \
              		loss.item(), trainAcc))
        
    if VISDOM:
        vis.line(X=torch.Tensor([epoch + 1]), \
                     Y=torch.Tensor([trainAcc * 100]),  win='trainAcc',  update='append')
        vis.line(X=torch.Tensor([epoch + 1]),\
                     Y=torch.Tensor([loss]), win='trainLoss', update='append')

    testAcc = 0
    for _, (x, y) in enumerate(test_loader): 
        # gives batch data, normalize x when iterate train_loader
        if torch.cuda.is_available():
            batch_x = Variable(x).cuda()
            batch_y = Variable(y).cuda()
        else:
            batch_x = Variable(x)
            batch_y = Variable(y)

        output, last_layer = myLeNet(batch_x)      # LeNet output
        loss = loss_func(output, batch_y)   		# cross entropy loss
        _, correct_label = torch.max(output, 1)  # 输出预测概率最大的值和标签
        # print('label', correct_label)
        correct_num = (correct_label == batch_y).sum()
        testAcc += correct_num.item()  

    testAcc = testAcc / float(MNISTtestData.test_labels.size(0))

    print('----------------test: Epoch [%d/%d] Acc: %.4f' % (epoch + 1, EPOCH,  testAcc))
    
    if VISDOM:
        vis.line(X=torch.Tensor([epoch + 1]), \
                 Y=torch.Tensor([testAcc * 100]),  win='testAcc',  update='append')
        vis.line(X=torch.Tensor([epoch + 1]),\
                     Y=torch.Tensor([loss]), win='testLoss', update='append')
    if epoch % 25 == 0:
        torch.save(myLeNet, './ModelBackup/MNIST_lenet_model_%d_%f.pkl'%(epoch, testAcc))


