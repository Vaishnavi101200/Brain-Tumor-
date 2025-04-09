# Brain Tumor Analysis System

This project implements a comprehensive system for brain tumor analysis in two stages:
1. Brain Tumor Detection and Segmentation using NIfTI files
2. Prognostic Analysis using H5 files

## Project Structure
```
GAN & MV/
├── stage1/                    # Tumor Detection Stage
│   ├── data/
│   │   └── BraTS_NII/        # Raw NIfTI files
│   ├── models/               # GAN and segmentation models
│   ├── preprocessing/        # NIfTI data preprocessing
│   └── processed_data/       # Preprocessed NIfTI data
│
├── stage2/                    # Prognostic Analysis Stage
│   ├── data/
│   │   └── BraTS_H5/        # Raw H5 files
│   ├── models/               # Prognostic models
│   ├── preprocessing/        # H5 data preprocessing
│   └── processed_data/       # Preprocessed H5 data
│
└── app/                      # Web application
```

## Implementation Steps

### Stage 1: Tumor Detection and Segmentation

1. **Data Preparation**
   - Place NIfTI files in `stage1/data/BraTS_NII/` with structure:
     ```
     BraTS_NII/
     └── BraTS2021_00000/
         ├── flair.nii
         ├── seg.nii
         ├── t1.nii
         ├── t1ce.nii
         └── t2.nii
     ```

2. **Preprocessing**
   ```bash
   cd stage1/preprocessing
   python nii_preprocessor.py
   ```
   This will:
   - Organize NIfTI files by case
   - Apply preprocessing transformations
   - Split data into train/val/test sets

3. **Model Training**
   ```bash
   cd stage1/models
   python train_gan.py
   ```
   This will:
   - Train the GAN model for tumor detection
   - Save model checkpoints
   - Generate sample images

### Stage 2: Prognostic Analysis

1. **Data Preparation**
   - Place H5 files in `stage2/data/BraTS_H5/data/`

2. **Preprocessing**
   ```bash
   cd stage2/preprocessing
   python h5_preprocessor.py
   ```
   This will:
   - Organize H5 files by case
   - Apply preprocessing transformations
   - Split data into train/val/test sets

3. **Model Training**
   ```bash
   cd stage2/models
   python train_prognostic.py
   ```
   This will:
   - Train the prognostic model
   - Save model checkpoints
   - Generate predictions

### Web Application

1. **Setup**
   ```bash
   cd app
   python app.py
   ```

2. **Usage**
   - Access the web interface at `http://localhost:5000`
   - Upload NIfTI files for tumor detection
   - Upload H5 files for prognostic analysis
   - View results and visualizations

## Environment Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing Details

### Stage 1: NIfTI Files
- Intensity normalization
- Data augmentation (rotation, flipping, zooming)
- Slice extraction from 3D volumes
- Train/val/test split

### Stage 2: H5 Files
- Data normalization
- Resizing to standard dimensions
- Train/val/test split

## Model Architecture

### Stage 1: Tumor Detection
- U-Net based generator
- PatchGAN discriminator
- Combined adversarial and pixel-wise loss

### Stage 2: Prognostic Analysis
- Feature extraction from H5 data
- Deep learning model for prognosis prediction

## Web Application Features
- File upload interface
- Real-time processing
- Visualization of results
- Error handling and feedback

## Troubleshooting

1. **Data Issues**
   - Verify file formats and naming conventions
   - Check data integrity
   - Monitor preprocessing steps

2. **Model Training**
   - Monitor loss curves
   - Check for overfitting
   - Validate data augmentation

3. **Application**
   - Check server logs
   - Verify API responses
   - Test error handling

## Support
For issues or questions, please open an issue in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BraTS dataset for providing the medical imaging data
- PyTorch and MONAI for deep learning framework
- Flask for web application framework 