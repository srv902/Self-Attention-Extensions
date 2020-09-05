# an attempt on 2D self attention (SA) on images with positional encoding
# Goal is to extend 2D SA to 3D SA to work on videos. Although there is Non-Local NN for global attention, there is a need to see how temporal aspects of a video are 
# handled and efficiently represented using 3D SA. 

# Another implementation is by EPFL where an image of shape H x W x Dim is reshaped to HW x Dim and treated as a sentence of tokens with feature dimension Dim.
# This can be easily incorporated with existing SA based models like BERT or Transformers in general for quick prototyping of SA based models on images. 

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# dilation 1 is by default in the convolution layer operation. 
# dilation 2 expands the kernel mask by filling in extra dimension between rows and columns with value 0.
# CURRENTLY ADDING FOR SINGLE HEAD ATTENTION NOW
class SA2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1, padding=1, bias=False):
        super(SA2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size # local kernel width for computing Self Attention
        self.kernel = (self.kernel_size, self.kernel_size)
        self.stride = stride # what steps to skip both in row and column
        if dilation > 1 or groups > 1:
            raise NotImplementedError('Dilation and Group methods are not supported yet')
        self.bias = bias
        self.padding = padding # 0 is by default in conv and in Self Attention, it might not be needed as we are not shriking the input feature map spatially

        assert padding != 0
        assert stride != 0 

        # to define unfold functions
        self.unfold = nn.Unfold(kernel_size=self.kernel, stride=self.stride, dilation=dilation) #, padding=self.padding)
        self.unfold_q = nn.Unfold(kernel_size=(1,1), stride=self.stride, dilation=dilation) #, padding=self.padding)

        # to define positional encodings 

        # to define adaptive kernel size (How and Why?) 

        # define Q, K, V matrices
        self.key_w = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) #nn.Linear(in_channels, out_channels)
        self.query_w = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_w = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.init_weights()

    def forward(self, input):
        batch, channels, height, width = input.size()
        # pad the input tensor
        input_padded = F.pad(input,[self.padding, self.padding, self.padding, self.padding])
        #input_padded = input

        # get key, query and values from matrix multiplication with input tensor
        keys = self.key_w(input_padded)
        queries = self.query_w(input)
        values = self.value_w(input_padded)

        # print(f'Shape of key {keys.shape}, query {queries.shape}, values {values.shape} ')

        # For Reference
        #print('--------------')
        #k_out = keys.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        #print('After unfolding ', k_out.shape) # 2 x 64 x 4 x 4 x 3 x 3. 64 is the out_channels and 3x3 are the local window extracted into array of blocks
        #print('--------------')

        keys_block = self.unfold(keys)
        queries_block = self.unfold_q(queries)
        values_block = self.unfold(values)
        # print(f'Keys arranged in sequence of sliding windows with dims {keys_block.shape}, query {queries_block.shape}, values {values_block.shape} ')

        # Number of patches/blocks in keys and querys must match for elementwise multiplication to happen
        assert keys_block.size(-1) == queries_block.size(-1)

        num_patches = keys_block.size(-1)
        num_query_neighbors = self.kernel_size**2
        # print(f'Number of patches in the input tensor {num_patches}')
        # print(f'Number of neighbors for a query {num_query_neighbors} ')

        assert keys_block.size(1)%num_query_neighbors == 0

        queries_reshaped = queries_block.view(batch, num_patches, 1, -1)
        keys_reshaped = keys_block.view(batch, num_patches, num_query_neighbors, -1)
        values_reshaped = values_block.view(batch, num_patches, num_query_neighbors, -1)
        
        # print(f'keys shape before multiplication {keys_reshaped.shape} , queries shape {queries_reshaped.shape}, values shape {values_reshaped.shape}')

        output = keys_reshaped * queries_reshaped
        # print(f'Output shape once key and queries are multiplied {output.shape}')

        # sum across channel dimension for all the neighbors of a query variable/pixel/output_node
        out = output.sum(dim=-1)

        # normalize by square root of output dimension
        # out = None

        # softmax probability. Attention scores of the neighborhood elements for each query. 
        out = F.softmax(out, dim=2)

        # to verify softmax scores sum to 1


        # weight the values with attention scores and to organize back to the shape of the initial Unfold op output. Initial input dim is (2,3,4,4) and num out_channels=16
        value_out = out.unsqueeze(-2) * values_reshaped.permute(0, 1, 3, 2) # 2,16,16,9 as output shape
        # print(f'>>>>>>>>>>>>Output of attn score * value tensor {value_out.shape} <<<<<<<<<<<<<<<<<<<')
        weighted_val = value_out.sum(dim=-1) # sum the attn scores * neighborhood value to get a number for the query pixel
        # print(f'Before reshaping the final tensor {weighted_val.shape}')
        weighted_val = weighted_val.contiguous().view(batch, -1, width//self.stride, height//self.stride) # 2, 16, 16 as output shape similar to output shape of UnFold op
        # print(f'Shape of attention score x value tensor {weighted_val.shape}')

        # In fold operation, the overlapping elements are summed and the sliding windows are created with overlap, there will be bogus numbers in the final feature map.
        # fold the values back into the shape of feature map tensor similar to conv feature map | Params fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
        #out = F.fold(weighted_val, output_size=(width, height), kernel_size=self.kernel, dilation=self.dilation, padding=self.padding, stride=self.stride)
        out = weighted_val
        return out
    
    def init_weights(self):
        nn.init.xavier_normal_(self.key_w.weight)
        nn.init.xavier_normal_(self.query_w.weight)
        nn.init.xavier_normal_(self.value_w.weight)

# keys, querys and values will be taken from the same input feature map ..
# assuming few defaults for now like stride=1, kernel_size=3. To generalize later once the functionalities are tested
def test_mod(input):
    pass

# https://discuss.pytorch.org/t/custom-convolution-dot-product/14992/25
# https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

# In Paper Stand Alone Self Attention, the attention is only applied to conv3x3 inside the BottleNeck block. No mention of BasicBlock for computation
# For now only BottleNeck block is supported

if __name__ == '__main__':
    #x = torch.rand(1,3,224,224)
    x = torch.rand(2, 3, 4, 4)
    #x = torch.rand(2, 3, 224, 224)
    print(x.shape)

    #conv1 = nn.Conv2d(3, 16, 3)
    #conv_out = conv1(x)

    #exit()

    # only considers valid padding with respect to the kernel_size. Stride works in any case
    attn1 = SA2D(3, 16, kernel_size=3, padding=1, stride=1)
    #attn2 = SA2D(16, 32, kernel_size=3, padding=1)

    out = attn1(x)
    #out = attn2(out)

    if out is not None:
        print(f'Final Layer Output shape {out.shape} ')
    else:
        print("No output yet!")

    model = nn.Sequential(attn1)
    pred = model(x)

    '''
    TODO:
    1. Add positional embedding and check
    2. Add normalization to softmax calculation
    3. Add support for Multi Head Attention
    4. Add support for grouped Self Attention (if possible)
    5. Embed SA layer into ResNet models and test the performance with variety of vision datasets (small dataset first)
    6. Once satisfactory performance, extend to Video Dataset. 
    7. TBD
    '''


