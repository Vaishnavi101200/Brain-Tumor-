import numpy as np
import h5py
import cv2
import os
from typing import List, Dict
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import glob

class H5Preprocessor:
    def __init__(self, base_path: str = None):
        self.base_path = base_path
        
    def organize_files(self, source_dir: str) -> Dict[str, str]:
        """
        Organize H5 files by case.
        Returns dictionary: {case_id: file_path}
        """
        h5_files = {}
        h5_paths = glob.glob(os.path.join(source_dir, "*.h5"))
        
        for h5_path in h5_paths:
            case_id = os.path.basename(h5_path).split('.')[0]
            h5_files[case_id] = h5_path
            
        return h5_files

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
        print("Organizing H5 files...")
        h5_files = self.organize_files(self.base_path)
        
        # Split cases
        case_ids = list(h5_files.keys())
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
        self._copy_case_files(h5_files, train_cases, split_dirs['train'], 'training')
        self._copy_case_files(h5_files, val_cases, split_dirs['val'], 'validation')
        self._copy_case_files(h5_files, test_cases, split_dirs['test'], 'testing')

        print("\nDataset splitting completed!")
        self._print_split_statistics(split_dirs)

    def _copy_case_files(self, h5_files: Dict, case_ids: List[str], 
                        target_dir: str, desc: str):
        """Copy all files for given cases to target directory"""
        for case_id in tqdm(case_ids, desc=f'Copying {desc} cases'):
            # Create case directory
            case_dir = os.path.join(target_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            
            # Copy H5 file
            h5_path = h5_files[case_id]
            shutil.copy2(h5_path, os.path.join(case_dir, f"{case_id}.h5"))

    def _print_split_statistics(self, split_dirs: dict):
        """Print statistics about the split dataset"""
        print("\nDataset Split Statistics:")
        print("-" * 50)
        for split, directory in split_dirs.items():
            case_dirs = [d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d))]
            print(f"{split.capitalize()} set:")
            print(f"  - Number of cases: {len(case_dirs)}")
            print(f"  - Files per case: 1 (H5 file)")
            print("-" * 50)
    
    def load_h5_file(self, file_path: str) -> np.ndarray:
        """Load .h5 file and return numpy array"""
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['data'])
        return data
    
    def preprocess_h5(self, data: np.ndarray) -> np.ndarray:
        """Preprocess H5 data for prognostic analysis"""
        # Normalize data
        data = (data - data.mean()) / data.std()
        
        # Resize if needed
        if data.shape[0] != 256 or data.shape[1] != 256:
            data = cv2.resize(data, (256, 256))
            
        return data

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = H5Preprocessor(
        base_path="stage2/data/BraTS_H5/data"
    )
    
    # Set output directory
    output_base_dir = "stage2/processed_data"
    
    # Split the dataset
    preprocessor.split_dataset(
        output_base_dir=output_base_dir,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    ) 