import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from gan_model import GAN
from prognostic_model import PrognosticModel, PrognosticTrainer

class BraTSDataset(Dataset):
    def __init__(self, nii_dir, h5_dir, transform=None):
        self.nii_dir = nii_dir
        self.h5_dir = h5_dir
        self.transform = transform
        self.preprocessor = DataPreprocessor()
        
        # Get list of files
        self.nii_files = [f for f in os.listdir(nii_dir) if f.endswith('.nii')]
        self.h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
        
        # Sort files to ensure matching pairs
        self.nii_files.sort()
        self.h5_files.sort()
        
    def __len__(self):
        return min(len(self.nii_files), len(self.h5_files))
    
    def __getitem__(self, idx):
        nii_path = os.path.join(self.nii_dir, self.nii_files[idx])
        h5_path = os.path.join(self.h5_dir, self.h5_files[idx])
        
        # Load and preprocess NIfTI data
        nii_data = self.preprocessor.load_nii_file(nii_path)
        nii_data = self.preprocessor.normalize_nii(nii_data)
        nii_data = self.preprocessor.preprocess_nii(nii_data)
        
        # Load and preprocess H5 data
        h5_data = self.preprocessor.load_h5_file(h5_path)
        h5_data = self.preprocessor.preprocess_h5(h5_data)
        
        # Convert to tensors
        nii_tensor = torch.from_numpy(nii_data).float()
        h5_tensor = torch.from_numpy(h5_data).float()
        
        return nii_tensor, h5_tensor

def train_gan(dataloader, gan, num_epochs, device):
    gan.generator.train()
    gan.discriminator.train()
    
    g_losses = []
    d_losses = []
    pixel_losses = []
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Train GAN
            losses = gan.train_step(images, masks)
            
            g_losses.append(losses['g_loss'])
            d_losses.append(losses['d_loss'])
            pixel_losses.append(losses['pixel_loss'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'G Loss': f"{losses['g_loss']:.4f}",
                'D Loss': f"{losses['d_loss']:.4f}",
                'Pixel Loss': f"{losses['pixel_loss']:.4f}"
            })
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            gan.save_models(f'models/gan_checkpoint_epoch_{epoch+1}.pth')
            
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(pixel_losses, label='Pixel Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_losses.png')
        plt.close()

def train_prognostic(dataloader, prognostic_trainer, num_epochs, device):
    prognostic_trainer.model.train()
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Train prognostic model
            loss = prognostic_trainer.train_step(images, labels)
            epoch_loss += loss
            
            # Calculate accuracy
            outputs = prognostic_trainer.model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            
            # Update progress bar
            accuracy = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f"{loss:.4f}",
                'Accuracy': f"{accuracy:.2f}%"
            })
        
        # Save epoch statistics
        losses.append(epoch_loss / len(dataloader))
        accuracies.append(100 * correct / total)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            prognostic_trainer.save_model(f'models/prognostic_checkpoint_epoch_{epoch+1}.pth')
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.savefig('prognostic_training_curves.png')
        plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Dataset paths (update these with your actual paths)
    nii_dir = 'path/to/nii/files'
    h5_dir = 'path/to/h5/files'
    
    # Create dataset and dataloader
    dataset = BraTSDataset(nii_dir, h5_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    # Initialize models
    gan = GAN(device=device)
    prognostic_model = PrognosticModel()
    prognostic_trainer = PrognosticTrainer(prognostic_model, device=device)
    
    # Training parameters
    num_epochs = 50
    
    print("Training GAN model...")
    train_gan(dataloader, gan, num_epochs, device)
    
    print("\nTraining Prognostic model...")
    train_prognostic(dataloader, prognostic_trainer, num_epochs, device)
    
    # Save final models
    gan.save_models('models/gan_final.pth')
    prognostic_trainer.save_model('models/prognostic_final.pth')
    
    print("Training completed!")

if __name__ == '__main__':
    main() 