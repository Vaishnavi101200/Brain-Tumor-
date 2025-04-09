import torch
from torch.utils.data import DataLoader
from data_preprocessing import DataPreprocessor
from models.gan_model import GAN
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_gan(
    data_dir: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 100,
    save_interval: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Initialize data preprocessor and GAN
    preprocessor = DataPreprocessor()
    gan = GAN(device=device)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data = preprocessor.load_nii_files(os.path.join(data_dir, 'train'))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (real_images, ground_truth) in enumerate(pbar):
                # Train step
                loss_G, loss_D = gan.train_step(real_images, ground_truth)
                
                # Update progress bar
                epoch_loss_G += loss_G
                epoch_loss_D += loss_D
                pbar.set_postfix({
                    'G Loss': f'{loss_G:.4f}',
                    'D Loss': f'{loss_D:.4f}'
                })
        
        # Calculate average losses
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} - G Loss: {avg_loss_G:.4f}, D Loss: {avg_loss_D:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'gan_epoch_{epoch+1}.pth')
            gan.save_checkpoint(checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
            
            # Generate and save sample images
            with torch.no_grad():
                sample_images = next(iter(train_loader))[0][:4].to(device)
                generated_images = gan.predict(sample_images)
                
                # Plot and save samples
                fig, axes = plt.subplots(4, 2, figsize=(10, 20))
                for i in range(4):
                    axes[i, 0].imshow(sample_images[i, 0].cpu().numpy(), cmap='gray')
                    axes[i, 0].set_title('Input')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
                    axes[i, 1].set_title('Generated')
                    axes[i, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'samples', f'samples_epoch_{epoch+1}.png'))
                plt.close()

if __name__ == "__main__":
    # Set paths
    data_dir = "GAN & MV/processed_data"
    output_dir = "GAN & MV/models/gan"
    
    # Training parameters
    batch_size = 8
    num_epochs = 100
    save_interval = 10
    
    # Start training
    train_gan(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_interval=save_interval
    ) 