# model.py to create custom models for 2D computer vision

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
#from torchsummary import summary
from pytorch_model_summary import summary

# import custom models from resnet.py
from resnet import resnet18, resnet50

def model_getter(name='resnet50', pretrained=False, use_attention=False):
    if name == 'resnet50':
        return resnet50(pretrained=pretrained,  use_attention=use_attention)
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    else:
        raise ValueError('Incorrect model name provided!')

class Model(nn.Module):
    def __init__(self, model_name='resnet50', use_attention=False):
        super(Model, self).__init__()
        self.base = model_getter(model_name, use_attention=use_attention)
        self.model = nn.Sequential(self.base)

    def forward(self, input):
        out = self.model(input)
        return out

if __name__ == '__main__':
    bs, height, width = 2, 224, 224
    input = torch.randn(bs, 3, height, width)

    model_name = 'resnet50'
    resnet_att = Model(model_name=model_name, use_attention=True)
    res1 = resnet_att(input)

    # plain resnet model without attention layers
    resnet_no_att = models.resnet50(pretrained=False)
    res2 = resnet_no_att(input)

    # show res shape
    print(f'ResNet with Attention Output Shape {res1.shape}')
    print('Resnet with Attention Model Summary')
    
    #print(summary(resnet_att, input_size=(input.size()[1:]) ) )
    print(summary(resnet_att.base, torch.zeros((2, 3, 224, 224)), show_input=True))

    print(summary(resnet_no_att, torch.zeros((2, 3, 224, 224)), show_input=True))

    '''
    total_params = 0
    for name, value in resnet_att.named_parameters():
        num_params = torch.prod(torch.tensor(value.shape), 0)
        print(name, value.size(), num_params)
        total_params += num_params
    '''

        

