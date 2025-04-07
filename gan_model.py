import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Decoder
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)
        
        return torch.sigmoid(dec1)

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
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
    
    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion_GAN = nn.BCELoss()
        self.criterion_pixelwise = nn.L1Loss()
        
    def train_step(self, real_images, real_masks):
        # Move data to device
        real_images = real_images.to(self.device)
        real_masks = real_masks.to(self.device)
        
        # Adversarial ground truths
        valid = torch.ones((real_images.size(0), 1, 1, 1), device=self.device)
        fake = torch.zeros((real_images.size(0), 1, 1, 1), device=self.device)
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate fake masks
        fake_masks = self.generator(real_images)
        
        # GAN loss
        pred_fake = self.discriminator(torch.cat((real_images, fake_masks), 1))
        loss_GAN = self.criterion_GAN(pred_fake, valid)
        
        # Pixel-wise loss
        loss_pixel = self.criterion_pixelwise(fake_masks, real_masks)
        
        # Total loss
        loss_G = loss_GAN + 100 * loss_pixel
        loss_G.backward()
        self.optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(torch.cat((real_images, real_masks), 1))
        loss_real = self.criterion_GAN(pred_real, valid)
        
        # Fake loss
        pred_fake = self.discriminator(torch.cat((real_images, fake_masks.detach()), 1))
        loss_fake = self.criterion_GAN(pred_fake, fake)
        
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': loss_G.item(),
            'd_loss': loss_D.item(),
            'pixel_loss': loss_pixel.item()
        }
    
    def predict(self, images):
        self.generator.eval()
        with torch.no_grad():
            images = images.to(self.device)
            fake_masks = self.generator(images)
        return fake_masks
    
    def save_models(self, path):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }, path)
    
    def load_models(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D']) 