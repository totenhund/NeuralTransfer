from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy


class ImageLoader:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 512
        self.loader = transforms.Compose(
            [transforms.Resize(self.img_size),
             transforms.ToTensor()])
        self.style_img = self.load_image("reference.jpg")
        self.content_img = self.load_image("input.jpg")
        self.unloader = transforms.ToPILImage()

    def load_image(self, img_name):
        image = Image.open(img_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def show_image(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def show_images(self):
        plt.ion()
        plt.figure()
        self.show_image(self.style_img, title='Style Image')
        plt.figure()
        self.show_image(self.content_img, title='Content Image')

