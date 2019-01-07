'''-----------------------------------------------------------
                            \\\||///
                           { -    - }
	             [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author		:		Yopen Hu
foldername  :		MNIST_train_LeNet
filename    :       MNISTLeNetConstruction.py
Description :		object segmentation in CIFAR10 using 
					CNN --- LeNet 
Date  	  By   Version          Change Description
==============================================================
19/01/04  HYP    0.0         extract data & read data
19/01/04  HYP    1.0       LeNet construction & utilize
--------------------------------------------------------------
                                    0ooo
--------------------------ooo0-----(    )---------------------
                         (    )     )  /
                          \  (     (__/
                           \__)
-----------------------------------------------------------'''
# 参考链接: https://blog.csdn.net/Gavinmiaoc/article/details/80492789
# standard library
import os 
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data 		# dataset
import torchvision						# torch vision library
import matplotlib.pyplot as plt 		# plot 2D data, Visualization
# from matplotlib import cm               # colormap 
import visdom # python -m visdom.server


torch.manual_seed(1)    # reproducible 设定生成随机数的种子，并返回一个 torch._C.Generator 对象

# Hyper Parameters 超参数
EPOCH = 250            # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 4000
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False  # 是否下载

##############################################################################################
########################### Mnist digits dataset preparation #################################
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
 
train_data = torchvision.datasets.MNIST(            # data size (60000, 28, 28) + label(60000)
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST
)

# # plot one example
# plt.imshow(train_data.train_data[100].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[100])
# plt.show()

# Data Loader for easy mini-batch return in training,
# the image batch shape will be (BATCH_SIZE, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
 
# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000] / 255.   
                              # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# print(test_x.size())        # torch.Size([2000, 1, 28, 28])
test_y = test_data.test_labels[:2000]

##############################################################################################
######################## define LeNet construction & Parameters ##############################
# widith_output = (widith_input + 2 * padding - kernel_size) / stride + 1
# padding = (kernel_size - 1) / 2 if stride = 1
class LeNet(nn.Module):
    def __init__(self):     
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels = 1,            # input height 输入通道
                out_channels = 6,           # n_filters 输出通道
                kernel_size = 5,            # filter size 卷积核
                stride = 1,                 # filter movement/step 步长
                padding = 2,                # if want same width and length of this image after con2d
            ),                              # output shape (6, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(                   # choose max value in 2x2 area
                kernel_size = 2,
                stride = 2
            ),                              # output shape (6, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Conv2d(6, 16, 5),            # output shape (16, 10, 10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2, 2),             # output shape (16, 5, 5)
        )
        self.fc1 = nn.Sequential( 
            nn.Linear(16 * 5 * 5, 120),     # fully connected layer, output 120 classes
            nn.ReLU() 
        ) 
        self.fc2 = nn.Sequential( 
            nn.Linear(120, 84),             # fully connected layer, output 84 classes
            nn.ReLU() 
        ) 
        self.fc3 = nn.Linear(84, 10)        # fully connected layer, output 10 classes

    # define forward propagation progress 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output, x            # return x for visualization

myLeNet = LeNet()
# print(myLeNet)    # net architecture

if torch.cuda.is_available():
    myLeNet.cuda()
    print("cuda is available, and the calculation will be moved to GPU\n")
else:
    print("cuda is unavailable!")

optimizer = torch.optim.Adam(myLeNet.parameters(), lr = LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()           # 交叉熵loss the target label is not one-hotted

##############################################################################################
###########################       training and testing       #################################
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   
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
 
        if (step + 1) % 3 == 0:
            _, correct_label = torch.max(output, 1)  # 输出预测概率最大的值和标签
            # print('label', correct_label)
            correct_num = (correct_label == batch_y).sum()

            acc = correct_num.item() / float(batch_y.size(0))

            print('Epoch [%d/%d], Iter [%2d/%d] Loss: %.4f, Acc: %.4f' % \
                  (epoch + 1, EPOCH, step + 1, len(train_data) // BATCH_SIZE, loss.item(), acc))
            
torch.save(myLeNet, 'MNIST_lenet_model.pkl')


            # if torch.cuda.is_available():
            #     test_x = test_x.cuda()
            #     test_y = test_y.cuda()
            # test_output, last_layer = myLeNet(test_x)
            # pred_y = torch.max(test_output, 1)[1].data.squeeze()    # 输出预测概率最大的值和标签
            # accuracy = sum(pred_y == test_y).item() / float(test_y.size(0))
            # print('testset -> step:',step // 50 + 1,' Epoch: ', epoch + 1, \
            #     '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)