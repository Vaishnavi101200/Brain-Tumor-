import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Generator(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self._block(input_channels, 64, batch_norm=False)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        self.enc5 = self._block(512, 1024)
        
        # Decoder
        self.dec5 = self._block(1024, 512)
        self.dec4 = self._block(1024, 256)
        self.dec3 = self._block(512, 128)
        self.dec2 = self._block(256, 64)
        self.dec1 = nn.Conv2d(128, output_channels, kernel_size=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_channels: int, out_channels: int, batch_norm: bool = True) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool(enc1))
        enc3 = self.enc3(self.maxpool(enc2))
        enc4 = self.enc4(self.maxpool(enc3))
        enc5 = self.enc5(self.maxpool(enc4))
        
        # Decoder
        dec5 = self.dec5(self.upsample(enc5))
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        
        return torch.sigmoid(dec1)

class Discriminator(nn.Module):
    def __init__(self, input_channels: int = 2):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class GAN:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion_GAN = nn.BCELoss()
        self.criterion_pixelwise = nn.L1Loss()
        
        self.lambda_pixel = 100
        
    def train_step(self, real_images: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[float, float]:
        # Move tensors to device
        real_images = real_images.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Adversarial ground truths
        valid = torch.ones((real_images.size(0), 1, 1, 1), device=self.device)
        fake = torch.zeros((real_images.size(0), 1, 1, 1), device=self.device)
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        fake_images = self.generator(real_images)
        
        # GAN loss
        pred_fake = self.discriminator(torch.cat((real_images, fake_images), 1))
        loss_GAN = self.criterion_GAN(pred_fake, valid)
        
        # Pixel-wise loss
        loss_pixel = self.criterion_pixelwise(fake_images, ground_truth)
        
        # Total loss
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(torch.cat((real_images, ground_truth), 1))
        loss_real = self.criterion_GAN(pred_real, valid)
        
        # Fake loss
        pred_fake = self.discriminator(torch.cat((real_images, fake_images.detach()), 1))
        loss_fake = self.criterion_GAN(pred_fake, fake)
        
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return loss_G.item(), loss_D.item()
    
    def save_checkpoint(self, path: str):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            return self.generator(image.to(self.device)) 