## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully Connected Layers
        # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs
        
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000) 
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 500)
        self.fc3 = nn.Linear(in_features = 500,    out_features = 136) 
        
        

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        # 1 Convolution + Activation + Pooling + Dropout layer
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        

        # 2 Convolution + Activation + Pooling + Dropout layer
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        

        # 3 Convolution + Activation + Pooling + Dropout layer
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
      

        # 4 Convolution + Activation + Pooling + Dropout layer
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        

        # Flattening the layer
        x = x.view(x.size(0), -1)
       

        # 1- Dense + Activation + Dropout layer
        x = self.drop5(F.relu(self.fc1(x)))
        

        # 2 - Dense + Activation + Dropout layer
        x = self.drop6(F.relu(self.fc2(x)))
        

        # Final Dense Layer
        x = self.fc3(x)
        
       
        return x
