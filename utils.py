import numpy as np
import torch
import cv2
import os
import torch.nn as nn

list_dir = os.listdir('./archive/img_align_celeba')

img = cv2.imread('./archive/img_align_celeba/' + list_dir[0])
img = cv2.resize(img, (178,178))
img = torch.tensor(img/127.5 -1).float()
img = torch.transpose(img,0,2)
img = torch.transpose(img,1,2).unsqueeze(dim = 0)

filter_size=3

print(img.shape)

img = nn.Conv2d(in_channels= 3,out_channels=8, kernel_size = filter_size)(img)
print(img.shape)
img = nn.Conv2d(in_channels= 8,out_channels=8, kernel_size = filter_size, stride = 2)(img)
print(img.shape)

img = nn.Conv2d(in_channels= 8, out_channels=16,kernel_size = filter_size)(img)
print(img.shape)
img = nn.Conv2d(in_channels= 16, out_channels=16, kernel_size =filter_size, stride = 2)(img)
print(img.shape)

img = nn.Conv2d(in_channels= 16, out_channels=32, kernel_size =filter_size)(img)
print(img.shape)
img = nn.Conv2d(in_channels= 32, out_channels=32,kernel_size = filter_size, stride = 2)(img)
print(img.shape)

img = nn.Conv2d(in_channels= 32, out_channels=64,kernel_size = filter_size)(img)
print(img.shape)
img = nn.Conv2d(in_channels= 64, out_channels=64,kernel_size = filter_size, stride = 2)(img)
print(img.shape)


latent = torch.tensor(np.random.uniform(0,1,(1,64))).float()
print(latent.shape)
latent = nn.Linear(64,8*8*64)(latent).reshape(1,64,8,8)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 64, out_channels = 32,kernel_size = filter_size, stride = 2)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 32, out_channels = 32,kernel_size = filter_size)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 32, out_channels = 16,kernel_size = filter_size, stride = 2)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 16, out_channels = 16,kernel_size = filter_size)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 16, out_channels = 8,kernel_size = filter_size, stride = 2)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 8, out_channels = 8,kernel_size = filter_size)(latent)
print(latent.shape)
latent = nn.ConvTranspose2d(in_channels = 8, out_channels = 3,kernel_size = filter_size, stride = 2)(latent)
print(latent.shape)

