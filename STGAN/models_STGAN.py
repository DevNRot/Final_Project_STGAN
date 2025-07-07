# Written by Devorah Rotman 316472026 and Carmit Kaye 346038169


import torch
import torch.nn as nn
from STGAN.custom_layers_STGAN import GenSynthesisBlock, Dis_block, EqualizedLinear


class Generator(nn.Module):
    def __init__(self, in_style, channels):
        super().__init__()
        self.in_style = in_style
        self.channels = channels
        # Learnable per-image style vector (optional)
        #self.w = nn.Parameter(torch.randn(1, in_style)  * 2)
        #second option for w: w as a function of the input image
        #self.style_mapper = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)),nn.Flatten(),EqualizedLinear(512 * 4 * 4, 512))

        # Generator blocks and toRGB layers
        self.gen_blocks = nn.ModuleList([GenSynthesisBlock(in_style, channels[0], channels[1], first_block=True)])

        self.num_of_blocks = 1

    def forward(self, x, alpha=1):
        #batch_size = x.size(0)
        #init_feats = x.clone()
        #w = self.w.repeat(batch_size, 1) #for original w
        #w = self.style_mapper(init_feats.clone())  # per-sample w

        if self.num_of_blocks > 1:
            for i, block in enumerate(self.gen_blocks[:-1]):
                #use_cond = (0< i < 5)
                x = block(x)
                # x is now the output of the second-last block
            upsampled = self.gen_blocks[-2].to_rgb(self.gen_blocks[-2].up_sample(x))

            # Feed x into the last block
            x = self.gen_blocks[-1](x)
            out = alpha * self.gen_blocks[-1].to_rgb(x) + (1 - alpha) * upsampled
        else:
            out = self.gen_blocks[-1].to_rgb(self.gen_blocks[-1](x))



        return torch.tanh(out)
        #return out
        #return torch.sigmoid(out)

    def grow(self):
        self.gen_blocks.append(GenSynthesisBlock(self.in_style, self.channels[self.num_of_blocks], self.channels[self.num_of_blocks + 1], first_block=False))
        self.num_of_blocks += 1


class Discriminator(nn.Module):
    def __init__(self, channels):
      super().__init__()
      self.channels   = channels
      self.dis_blocks = nn.ModuleList([Dis_block(channels[1], channels[0], last_block = True)])
      self.num_of_blocks = 1
    def forward(self, x, alpha = 0.2):
      input = x
      x = self.dis_blocks[-1].from_rgb(x)
      x = self.dis_blocks[-1](x)
      if self.num_of_blocks > 1:
        down_sampled_rgb = self.dis_blocks[-1].down_sample(input)
        down_sampled = self.dis_blocks[-2].from_rgb(down_sampled_rgb)
        x = alpha * x + (1 - alpha) * down_sampled
        for block_i in reversed(range(0, self.num_of_blocks - 1)):
            x = self.dis_blocks[block_i](x)
      return x
    def grow(self,):
      self.dis_blocks.append(Dis_block(self.channels[self.num_of_blocks + 1], self.channels[self.num_of_blocks]))
      self.num_of_blocks += 1
    
