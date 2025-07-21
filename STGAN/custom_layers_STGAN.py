# Written by Devorah Rotman and Carmit Kaye

# some codes copied from https://github.com/nashory/pggan-pytorch

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# for equaliaeed-learning rate.
class EqualizedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(c_in, c_out, k_size, stride, pad)

        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, x):
        return self.conv(x)

class EqualizedLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super(EqualizedLinear, self).__init__()
        linear = nn.Linear(c_in, c_out)

        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)

def pixel_norm(z):
  return z / (torch.mean(z**2,dim = 1, keepdim = True) + 1e-7) ** 0.5

def minibatch_stddev_layer(x, group_size = 4, num_new_features = 1):
  group_size = min(group_size, x.shape[0])
  y = x.reshape(group_size, -1, num_new_features, x.shape[1] //num_new_features, x.shape[2], x.shape[3])
  y = torch.sqrt(torch.mean((y - torch.mean(y, dim = 0, keepdim = True)) ** 2, dim = 0) + 1e-8) ## calc std
  y = torch.mean(y, dim =[2,3,4], keepdim = True)
  y = torch.squeeze(y, dim=2)
  y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
  return torch.cat([x,y],dim = 1)


#below is the simplified GenSynthesisBlock
class GenSynthesisBlock(nn.Module):
    def __init__(self, in_style, in_block, block_channels, first_block=False):
        super().__init__()
        self.first_block = first_block
        if not first_block:
            self.first_conv = EqualizedConv2d(in_block, block_channels, 3, 1, 1)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu_1 = nn.LeakyReLU(0.2)
        self.conv = EqualizedConv2d(block_channels, block_channels, 3, 1, 1)
        self.lrelu_2 = nn.LeakyReLU(0.2)
        self.to_rgb = EqualizedConv2d(block_channels, 3, 1, 1, 0)

    def forward(self, x):
        if not self.first_block:
            x = self.up_sample(x)
            x = self.first_conv(x)

        x = self.lrelu_1(x)
        x = self.conv(x)
        x = self.lrelu_2(x)

        return x


class Dis_block(nn.Module):
  def __init__(self,in_block, block_channels, last_block = False):
    super().__init__()
    self.down_sample = nn.Upsample(scale_factor = 0.5, mode = 'bilinear', align_corners=False)
    self.last_block  = last_block
   
    if last_block:
        self.block = nn.Sequential(
            EqualizedConv2d(in_block + 1, block_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(block_channels, block_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),       # Shrinks spatial dims to 1x1
            nn.Flatten(),                       #  So we now flatten only [B, C]
            EqualizedLinear(block_channels, 1)  #  Matches correct input size
        )
   
    else:
        self.block = nn.Sequential(
                   EqualizedConv2d(in_block, block_channels, 3, 1, 1),
                   nn.LeakyReLU(0.2),
                   EqualizedConv2d(block_channels, block_channels, 3, 1, 1),
                   nn.LeakyReLU(0.2),
                    )
    self.from_rgb = EqualizedConv2d(3, in_block, 1, 1, 0)
    
  def forward(self, x):
    if self.last_block:
      x = minibatch_stddev_layer(x)

    x = self.block(x)

    if not self.last_block:
      x = self.down_sample(x)
    return x

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

