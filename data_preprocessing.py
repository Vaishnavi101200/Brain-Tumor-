import numpy as np
import nibabel as nib
import h5py
import cv2
from monai.transforms import (
    Compose, LoadImage, AddChannel, ScaleIntensity,
    RandRotate90, RandFlip, RandZoom, RandGaussianNoise
)
import os
from typing import Tuple, List
import torch

class DataPreprocessor:
    def __init__(self, nii_path: str = None, h5_path: str = None):
        self.nii_path = nii_path
        self.h5_path = h5_path
        
    def load_nii_file(self, file_path: str) -> np.ndarray:
        """Load .nii file and return numpy array"""
        img = nib.load(file_path)
        return img.get_fdata()
    
    def load_h5_file(self, file_path: str) -> np.ndarray:
        """Load .h5 file and return numpy array"""
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['data'])
        return data
    
    def normalize_nii(self, data: np.ndarray) -> np.ndarray:
        """Normalize NIfTI data to [0, 1] range"""
        return (data - data.min()) / (data.max() - data.min())
    
    def preprocess_nii(self, data: np.ndarray) -> np.ndarray:
        """Preprocess NIfTI data with MONAI transforms"""
        transforms = Compose([
            AddChannel(),
            ScaleIntensity(),
            RandRotate90(prob=0.5),
            RandFlip(prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandGaussianNoise(prob=0.5, std=0.1)
        ])
        
        # Convert to torch tensor for transforms
        data_tensor = torch.from_numpy(data).float()
        transformed = transforms(data_tensor)
        return transformed.numpy()
    
    def extract_slices(self, volume: np.ndarray, num_slices: int = 3) -> List[np.ndarray]:
        """Extract middle slices from 3D volume"""
        middle_idx = volume.shape[2] // 2
        start_idx = middle_idx - num_slices // 2
        end_idx = middle_idx + num_slices // 2 + 1
        return [volume[:, :, i] for i in range(start_idx, end_idx)]
    
    def preprocess_h5(self, data: np.ndarray) -> np.ndarray:
        """Preprocess H5 data for prognostic analysis"""
        # Normalize data
        data = (data - data.mean()) / data.std()
        
        # Resize if needed
        if data.shape[0] != 256 or data.shape[1] != 256:
            data = cv2.resize(data, (256, 256))
            
        return data
    
    def prepare_batch(self, nii_files: List[str], h5_files: List[str], 
                     batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare batch of data for training"""
        nii_batch = []
        h5_batch = []
        
        for nii_file, h5_file in zip(nii_files[:batch_size], h5_files[:batch_size]):
            # Process NIfTI data
            nii_data = self.load_nii_file(nii_file)
            nii_data = self.normalize_nii(nii_data)
            nii_data = self.preprocess_nii(nii_data)
            nii_batch.append(nii_data)
            
            # Process H5 data
            h5_data = self.load_h5_file(h5_file)
            h5_data = self.preprocess_h5(h5_data)
            h5_batch.append(h5_data)
            
        return np.array(nii_batch), np.array(h5_batch) 