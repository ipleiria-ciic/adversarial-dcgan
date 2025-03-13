import random

import torch
import torch.nn as nn
import torchvision.datasets as TD
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

def use_device():
    """
    Gets the device available to use while using Torch.

    Return:
    - device: Device (CPU or GPU).
    """
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def prepare_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def get_dataloader(dataroot, image_size, batch_size, workers):
    """
    Prepare a dataloader for a given dataroot.

    Parameters:
    - dataroot: Path to the dataset.
    - image_size: Image size.
    - batch_size: Batch size.
    - workers: Number of workers.

    Return:
    - dataloader: Dataloader for a given dataset.
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TD.ImageFolder(root=dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader

def weights_init(m):
    """
    Initialise weights for a given class.

    Parameters:
    - m: A given class.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Generator class.
    """
    def __init__(self, gpu_number, feature_map_g, channel_number, latent_vector_z):
        super(Generator, self).__init__()
        self.gpu_number = gpu_number
        self.feature_map_g = feature_map_g
        self.channel_number = channel_number
        self.latent_vector_z = latent_vector_z
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_z, feature_map_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_g*8, feature_map_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_g*4, feature_map_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_g*2, feature_map_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_g, channel_number, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    """
    Discriminator class.
    """
    def __init__(self, gpu_number, feature_map_d, channel_number, latent_vector_z):
        super(Discriminator, self).__init__()
        self.gpu_number = gpu_number
        self.feature_map_d = feature_map_d
        self.channel_number = channel_number
        self.latent_vector_z = latent_vector_z
        self.main = nn.Sequential(
            nn.Conv2d(channel_number, feature_map_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_d, feature_map_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_d*2, feature_map_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_d*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_d*4, feature_map_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_d*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_map_d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
class Encoder(nn.Module):
    """
    Encoder class.
    """
    def __init__(self, feature_map_e, channel_number, latent_vector_z):
        super(Encoder, self).__init__()
        self.feature_map_e = feature_map_e
        self.channel_number = channel_number
        self.latent_vector_z = latent_vector_z
        self.main = nn.Sequential(
            nn.Conv2d(channel_number, feature_map_e, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_e),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_e, feature_map_e*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_e*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_e*2, feature_map_e*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_e*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_e*4, feature_map_e*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_e*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_e*8, latent_vector_z, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer=8):
        """
        Uses VGG-19 to compute perceptual loss.
        The layer parameter determines which VGG feature map is used.
        """
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:layer])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
    def forward(self, x, y):
        x_vgg = self.vgg_layers(x) 
        y_vgg = self.vgg_layers(y)
        loss = nn.functional.mse_loss(x_vgg, y_vgg)
        return loss