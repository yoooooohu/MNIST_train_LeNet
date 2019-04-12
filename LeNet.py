import torch.nn as nn

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