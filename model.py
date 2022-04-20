import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from logger import Logger
import cv2
import os
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, filter_size = 3, lr = 3e-4):
        super(Discriminator,self).__init__()

        self.net = nn.Sequential()

        self.net.append(nn.Conv2d(in_channels= 3,out_channels=8,kernel_size = filter_size))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.Conv2d(in_channels= 8,out_channels=8,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.LeakyReLU(0.3))

        self.net.append(nn.Conv2d(in_channels= 8, out_channels=16,kernel_size = filter_size))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.Conv2d(in_channels= 16, out_channels=16, kernel_size =filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.LeakyReLU(0.3))

        self.net.append(nn.Conv2d(in_channels= 16, out_channels=32,kernel_size = filter_size))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.Conv2d(in_channels= 32, out_channels=32,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.LeakyReLU(0.3))

        self.net.append(nn.Conv2d(in_channels= 32, out_channels=64,kernel_size = filter_size))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.Conv2d(in_channels= 64, out_channels=64, kernel_size =filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.LeakyReLU(0.3))

        self.classifier = nn.Linear( 64*8*8 ,1)
        
        self.optimizer = Adam(self.parameters(),lr = lr)
    
    def forward(self, x):
        
        x = self.net(x)
        x = self.classifier(torch.flatten(x, start_dim = 1))
        return F.sigmoid(x)
     

class Generator(nn.Module):
    def __init__(self, filter_size = 3, latent_dim = 64, lr = lr):
        super(Generator,self).__init__()
        
        self.initial =  nn.Linear(64,))

        self.net = nn.Sequential()
                
        self.net.append(nn.ConvTranspose2d(in_channels = 64, out_channels = 32,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(n.LeakyReLU(0.3))

        self.net.append(nn.ConvTranspose2d(in_channels = 32, out_channels = 16,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(n.LeakyReLU(0.3))

        self.net.append(nn.ConvTranspose2d(in_channels = 16, out_channels = 8,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(n.LeakyReLU(0.3))

        self.net.append(nn.ConvTranspose2d(in_channels = 8, out_channels = 3,kernel_size = filter_size, stride = 2))
        self.net.append(nn.BatchNorm2d())
        self.net.append(nn.Tanh())
    
        self.optimizer = Adam(self.parameters(),lr = lr)    
    
    def forward(self, x):
        x = self.initial(x).reshape()
        return self.net(x)


class Imageset(Dataset):
      def __init__(self, data):
          self.data = torch.tensor(data).float()
      def __get_item(self,index):
          return self.data[index]
      def __len__(self):
          return len(self.data)
      
        
class GAN:
    def __init__(self, data_dir, log_dir, filter_size, latent_dim, spatial_dim, device = torch.device('cpu')):
        
        self.device = device
        
        self.gen = Generator(filter_size = filter_size, latent_dim = latent_dim)
        self.gen.to(self.device)
        self.discri = Dicriminator(filter_size = filter_size)
        self.discri.to(self.device)
        
        self.logger = Logger(log_dir)
        
        self.data_dir = data_dir
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim 
        
    def load_img(self, size):
        
        file_list = os.listdir(self.data_dir)
        images = np.empty((size, self.spatial_dim, self.spatial_dim, 3))
        for i in range(size):
            img = cv2.imread(file_list[i])
            img = cv2.resize(img,(self.spatial_dim, self.spatial_dim))
            img = np.astype(np.float32)/127.5 - 1
            images[i] = img
        
        return Imagesset(images)
            
            
    def train_discri(self, size = int(5e4)):
        
        loader = DataLoader(load_img(size), batch_size = 32)
        
        discri_loss = torch.tensor(0.0)
        gen_loss = torch.tensor(0.0)
        
        for i,sample in enumerate(loader):
            discri_loss += torch.log(self.discri(sample))
            
        for i in range(size//32):
            latent_vec = torch.tensor(np.random.uniform(32,self.latent_size))
            with torch.no_grad():
              gen_img = self.gen(latent_vec)
            
            gen_loss += torch.log(1-self.discri(gen_img))
            
        self.logger.add_scalar('discri loss discri', -discri_loss.item())
        self.logger.add_scalar('discri loss gen', -gen_loss.item())
        self.logger.add_scalar('discri loss', -(discri_loss + gen_loss).item())
        
        return -(discri_loss + gen_loss)
    
    
    def train_gen(self, size = int(5e4)):
        
        gen_loss = torch.tensor(0.0)

        for i in range(size//32):
            latent_vec = torch.tensor(np.random.uniform(0,1,(32,self.latent_size)))
            gen_img = self.gen(latent_vec)
            with torch.no_grad():
              prob = self.discri(gen_img)
            gen_loss += torch.log(1-prob)
    
        self.logger.add_scalar('gen loss ', gen_loss.item())
        return gen_loss
    
    def train(self, epochs = 1, discri_epochs = 1, size = int(5e4)):
        
        self.discri.train(True)
        self.gen.train(True)
        
        for _ in range(epochs):
            for _ in range(discri_epochs):
                loss = self.train_discri(size)
                self.discri.optimizer.zero_grad()
                loss.backward()
                self.discri.optimizer.step()
            
            loss = self.train_gen(size)
            self.gen.optimizer.zero_grad()
            loss.backward()
            self.gen.optimizer.step()
        
        self.discri.train(False)
        self.gen.train(False)
    
    def generate_image(self, n_image):
        
        latent_vec = torch.tensor(np.random.uniform(0,1,(n_images,self.latent_size)))        
        with torch.no_grad():
            gen_img = self.gen(latent_vec)
        
        return gen_img
                
        
        
            
            
        
        