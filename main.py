import torch
from model import GAN
from logger import Logger
from config import Config
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gan = GAN(Config['data_dir'], Config['log_dir'], Config['filter_size'], Config['latent_dim'], Config['spatial_dim'], device = device)

gan.train( epochs = Config['epochs'], discri_epochs = Config['discri-epochs'], size = Config['size'])