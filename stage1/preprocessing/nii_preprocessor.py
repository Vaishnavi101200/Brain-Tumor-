import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImage, AddChannel, ScaleIntensity,
    RandRotate90, RandFlip, RandZoom, RandGaussianNoise
)
import os
from typing import Tuple, List, Dict
import torch
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import glob

class NIIPreprocessor:
    def __init__(self, base_path: str = None):
        self.base_path = base_path
        
    def organize_files(self, source_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Organize NIfTI files by case and modality.
        Returns dictionary: {case_id: {modality: file_path}}
        """
        cases = {}
        case_dirs = glob.glob(os.path.join(source_dir, "BraTS*"))
        
        for case_dir in case_dirs:
            case_id = os.path.basename(case_dir)
            cases[case_id] = {}
            
            # Find all .nii files in the case directory
            nii_files = glob.glob(os.path.join(case_dir, "*.nii"))
            
            for nii_file in nii_files:
                modality = os.path.basename(nii_file).split('.')[0]  # flair, t1, t1ce, t2, seg
                cases[case_id][modality] = nii_file
                
        return cases

    def split_dataset(self, output_base_dir: str, 
                     train_size: float = 0.7, val_size: float = 0.15,
                     test_size: float = 0.15, random_state: int = 42):
        """
        Split the dataset into training, validation, and testing sets.
        """
        # Create output directories
        splits = ['train', 'val', 'test']
        split_dirs = {}
        for split in splits:
            split_dir = os.path.join(output_base_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            split_dirs[split] = split_dir

        # Organize files
        print("Organizing NIfTI files...")
        cases = self.organize_files(self.base_path)
        
        # Split cases
        case_ids = list(cases.keys())
        train_cases, temp_cases = train_test_split(
            case_ids, train_size=train_size, 
            random_state=random_state
        )
        
        val_ratio = val_size / (val_size + test_size)
        val_cases, test_cases = train_test_split(
            temp_cases, train_size=val_ratio,
            random_state=random_state
        )

        # Copy files to respective directories
        self._copy_case_files(cases, train_cases, split_dirs['train'], 'training')
        self._copy_case_files(cases, val_cases, split_dirs['val'], 'validation')
        self._copy_case_files(cases, test_cases, split_dirs['test'], 'testing')

        print("\nDataset splitting completed!")
        self._print_split_statistics(split_dirs)

    def _copy_case_files(self, cases: Dict, case_ids: List[str], 
                        target_dir: str, desc: str):
        """Copy all files for given cases to target directory"""
        for case_id in tqdm(case_ids, desc=f'Copying {desc} cases'):
            # Create case directory
            case_dir = os.path.join(target_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            
            # Copy NIfTI files
            for modality, nii_path in cases[case_id].items():
                shutil.copy2(nii_path, os.path.join(case_dir, f"{modality}.nii"))

    def _print_split_statistics(self, split_dirs: dict):
        """Print statistics about the split dataset"""
        print("\nDataset Split Statistics:")
        print("-" * 50)
        for split, directory in split_dirs.items():
            case_dirs = [d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d))]
            print(f"{split.capitalize()} set:")
            print(f"  - Number of cases: {len(case_dirs)}")
            print(f"  - Files per case: 5 (flair, t1, t1ce, t2, seg)")
            print("-" * 50)
        
    def load_nii_file(self, file_path: str) -> np.ndarray:
        """Load .nii file and return numpy array"""
        img = nib.load(file_path)
        return img.get_fdata()
    
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

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = NIIPreprocessor(
        base_path="stage1/data/BraTS_NII"
    )
    
    # Set output directory
    output_base_dir = "stage1/processed_data"
    
    # Split the dataset
    preprocessor.split_dataset(
        output_base_dir=output_base_dir,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    ) 