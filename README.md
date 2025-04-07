# Brain Tumor Detection and Analysis System

This project implements a comprehensive system for brain tumor detection, segmentation, and prognostic analysis using GANs and Machine Vision. The system can process both .nii and .h5 files from the BraTS dataset.

## Features

- Brain tumor detection and segmentation using GANs
- Prognostic analysis for treatment outcomes
- Web-based interface for easy interaction
- Support for both .nii and .h5 file formats
- Detailed tumor volume calculation
- Confidence-based prognosis prediction

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd brain-tumor-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir uploads
mkdir models
```

## Project Structure

- `data_preprocessing.py`: Handles data loading and preprocessing for both .nii and .h5 files
- `gan_model.py`: Implements the GAN architecture for tumor detection and segmentation
- `prognostic_model.py`: Contains the prognostic analysis model
- `app.py`: Flask web application for local deployment
- `templates/`: Contains HTML templates for the web interface
- `uploads/`: Temporary storage for uploaded files
- `models/`: Storage for trained model weights

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload either a .nii file for tumor detection or a .h5 file for prognostic analysis.

## Model Training

To train the models from scratch:

1. Prepare your dataset:
   - Place .nii files in a directory for tumor detection
   - Place .h5 files in a directory for prognostic analysis

2. Train the GAN model:
```python
from gan_model import GAN
from data_preprocessing import DataPreprocessor

# Initialize models and data preprocessor
gan = GAN()
preprocessor = DataPreprocessor()

# Load and preprocess data
# ... (implement training loop)
```

3. Train the prognostic model:
```python
from prognostic_model import PrognosticModel, PrognosticTrainer

# Initialize model and trainer
model = PrognosticModel()
trainer = PrognosticTrainer(model)

# Load and preprocess data
# ... (implement training loop)
```

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload and process medical images
  - Accepts .nii files for tumor detection
  - Accepts .h5 files for prognostic analysis

## Response Format

For .nii files:
```json
{
    "status": "success",
    "tumor_detected": true/false,
    "tumor_volume": float,
    "mask": [...]
}
```

For .h5 files:
```json
{
    "status": "success",
    "prognosis": "Good/Moderate/Poor",
    "confidence": float
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BraTS dataset for providing the medical imaging data
- PyTorch and MONAI for deep learning framework
- Flask for web application framework 