import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

from utils import *

# #ChanRev meth7    

class ConvBlock(nn.Module):
    def __init__(self, input_feat, output_feat, kernel_size, dropout_p = 0, masking = True, weight_norm = False):
        super(ConvBlock, self).__init__()       
        '''
        Single convolution block layer with skip connection

        args:
            input_feat: # input channels
            output_feat: # output channels
            kernel_size: size of 1d kernel
            dropout_p: dropout parameter
            masking: boolean, turns convolution masking on/off. Convolutions are masked to prevent access to 'future' information
        '''

        # convolution masking
        self.masking = masking
        if self.masking:
            padding = kernel_size - 1
        else:
            padding = kernel_size//2

        if weight_norm:
            self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = padding, stride = 1))
        else:
            self.conv = nn.Conv1d(in_channels = input_feat, out_channels = output_feat, kernel_size = kernel_size, padding = padding, stride = 1)

        self.dropout = nn.Dropout(p = dropout_p)
        self.kernel_size = kernel_size
        self.activation = torch.relu
        self.downsample = nn.Linear(input_feat, int(output_feat))
        self.bn = nn.BatchNorm1d(output_feat)

    def forward(self, x):
        identity = x   # skip connection
        x = self.conv(x) 
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
        if self.masking: x = x[:,:,:-(self.kernel_size-1)]  # truncate by kernel_size - 1
        if (not self.masking) and (self.kernel_size%2==0): x = x[:,:,:-1] # when kernel size is odd, truncate to ensure each layer is the same width

        # downsample the identity if there is a dimension mismatch
        if identity.shape != x.shape:
            identity = identity.transpose(1, 2)
            identity = self.downsample(identity)
            identity = identity.transpose(1, 2)

        return x + identity  


class CNN(nn.Module):
    def __init__(self, num_hidden_units, max_length, kernel_size, num_layers = 1, dropout_p = 0, num_linear_units = 10, num_output_feat = 128, conv_masking = False, weight_norm = False, append_transform_flag = False, **kwargs):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.max_length = max_length
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.num_linear_units = num_linear_units
        self.append_transform_flag = append_transform_flag

        if append_transform_flag:
            self.feature_padding = 3
        else:
            self.feature_padding = 0

        # Encoder ==========================================
        conv_layers = []
        for i in range(num_layers):  # define output channels for convolution operation. Note: Subsequent GLU downsamples features by 1/2
            if i == 0:
                conv_layers.append(ConvBlock(1, num_hidden_units, kernel_size, dropout_p, masking=conv_masking, weight_norm = weight_norm))
            else:
                conv_layers.append(ConvBlock(num_hidden_units, num_hidden_units, kernel_size, dropout_p, masking = conv_masking, weight_norm = weight_norm))

        self.conv_encode = nn.Sequential(*conv_layers)
        self.fc1 = nn.Linear(num_hidden_units, 10)
        self.fc2 = nn.Linear(int((self.max_length + self.feature_padding) * 10), num_linear_units)
        
        g_input_feat = num_linear_units
        self.g = nn.Sequential(
            nn.Linear(g_input_feat, g_input_feat//4, bias=False),
            nn.BatchNorm1d(g_input_feat//4),
            nn.ReLU(inplace=True),
            nn.Linear(g_input_feat//4, num_output_feat, bias=True)
            )
    
    def forward(self, x, aux = None):
        x = x.unsqueeze(1)
        x = self.conv_encode(x)
        x = x.transpose(1,2) # n x t x conv_feat

        x = torch.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        features = self.fc2(torch.flatten(x, 1))

        output = self.g(features)


        return F.normalize(features, dim = -1), F.normalize(output, dim = -1)


