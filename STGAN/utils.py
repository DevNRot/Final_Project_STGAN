#Written by Devorah Rotman 316472026 and Carmit Kaye 346038169


import torchvision.utils as vutils
import numpy as np
import random
import torch
import yaml
#import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
#import numpy as np
#import torchvision.utils as vutils


def visualize_output(fake_imgs, scale):
    plt.figure(figsize=(10, 10))
    grid_img = vutils.make_grid(fake_imgs.detach().cpu(), normalize=True)
    plt.imshow(np.transpose(grid_img, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"Samples/scale_{scale}.png")  # optional: organize into folder
    plt.close()
    
def visualize_graphs(generator_loss, discriminator_loss, scale):
    plt.figure(figsize=(10, 10))
    plt.plot(generator_loss)
    plt.plot(discriminator_loss)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend([f"generator loss", "discriminator loss"])
    plt.tight_layout()
    plt.savefig(f"graphs_scale_{scale}.png")  # optional: organize into folder
    plt.close()
"""    
def visualize_output(fake_imgs, scale):
  plt.figure(figsize=(10,10))
  plt.imshow(np.transpose(vutils.make_grid(fake_imgs.detach().cpu(), normalize=True),(1,2,0)))
  plt.savefig(f"Samples from scale {scale}.png") 
"""
def get_style_mixing(batch_size, in_style, device, num_layers = 1, prop = 0.8):
  style_mixing, z = (None, None) if random.random() > prop else (num_layers, torch.randn(batch_size, in_style, device = device))
  return style_mixing, z

def parse_yaml(path):
    with open(path, "r") as ya:
        try:
            config = yaml.safe_load(ya)
            return config
        except yaml.YAMLError as exc:
            print(exc)
def grow_test(gen, dis, loader):
    gen.grow()
    dis.grow()
    gen.cuda()
    dis.cuda()
    train_loader = loader.grow()
    
def load_ckpts_test(ckpt_path, gen, dis, loader):
    ckpt    = torch.load(ckpt_path + 'last.pt', map_location = 'cpu')
    for i in range(ckpt['grow_rank']):
        gen, dis, train_loader = grow(gen, dis, loader)
    gen.load_state_dict(ckpt['generator'])
    dis.load_state_dict(ckpt['discriminator'])
    return gen, dis, train_loader

