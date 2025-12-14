"""
Pix2Pix Model for Image-to-Image Translation of CT Scans to MRI Scans
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets

# Define the Generator Network which is a encoder-decoder architecture
class Generator(nn.Module):
    def __init__(self, in_channel: int = 3, 
                 out_channel: int = 3, 
                 basefeature: int = 64, 
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, basefeature, kernel_size, stride, padding), # no batchnorm on first layer (transformation already normalizes input)
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature, basefeature*2, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature*2, basefeature*4, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature*4, basefeature*8, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*8),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(basefeature*8, basefeature*4, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*4),
            nn.ReLU(),
            nn.ConvTranspose2d(basefeature*4, basefeature*2, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*2),
            nn.ReLU(),
            nn.ConvTranspose2d(basefeature*2, basefeature, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature),
            nn.ReLU(),
            nn.ConvTranspose2d(basefeature, out_channel, kernel_size, stride, padding), # no batchnorm on last layer
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
# Define the Discriminator Network which is a PatchGAN
class Discriminator(nn.Module):
    def __init__(self, in_channel: int = 6, 
                 basefeature: int = 64, 
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 out_channel: int = 1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, basefeature, kernel_size, stride, padding), # no batchnorm on first layer (transformation already normalizes input)
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature, basefeature*2, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature*2, basefeature*4, kernel_size, stride, padding),
            nn.BatchNorm2d(basefeature*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature*4, basefeature*8, kernel_size, 1, padding), # stride 1 for last deep layer to preserve spatial dimensions
            nn.BatchNorm2d(basefeature*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(basefeature*8, out_channel, kernel_size, 1, padding), # output single channel for real/fake classification
            nn.Sigmoid()
        )

    def forward(self, mri_image: torch.Tensor, ct_image: torch.Tensor) -> torch.Tensor:
        # Concatenate CT and MRI images along the channel dimension
        x = torch.cat([mri_image, ct_image], dim=1)
        valid = self.model(x)
        return valid
    
def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # initialize conv layers with normal distribution
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # initialize batchnorm layers with mean 1 and std 0.02
        nn.init.constant_(m.bias.data, 0)
        
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    



