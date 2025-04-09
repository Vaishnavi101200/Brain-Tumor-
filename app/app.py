from flask import Flask, render_template, request, jsonify
import os
import torch
from werkzeug.utils import secure_filename
import numpy as np
from models.gan_model import GAN
from data_preprocessing import DataPreprocessor
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'nii', 'h5'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = GAN(device=device)
preprocessor = DataPreprocessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if filename.endswith('.nii'):
                # Process NIfTI file
                result = process_nii_file(filepath)
            else:
                # Process H5 file
                result = process_h5_file(filepath)
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_nii_file(filepath):
    # Load and preprocess NIfTI file
    img = nib.load(filepath)
    data = img.get_fdata()
    
    # Preprocess data
    data = preprocessor.normalize_nii(data)
    data = preprocessor.preprocess_nii(data)
    
    # Convert to tensor
    data_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
    
    # Generate prediction
    with torch.no_grad():
        prediction = gan.predict(data_tensor)
    
    # Convert to numpy
    prediction = prediction.squeeze().cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(data.squeeze(), cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_base64 = plot_to_base64(fig)
    plt.close()
    
    return {
        'status': 'success',
        'plot': plot_base64,
        'message': 'Tumor detection completed successfully'
    }

def process_h5_file(filepath):
    # Load and preprocess H5 file
    with h5py.File(filepath, 'r') as f:
        data = np.array(f['data'])
    
    # Preprocess data
    data = preprocessor.preprocess_h5(data)
    
    # Create visualization
    plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap='gray')
    plt.title('Processed Data')
    plt.axis('off')
    
    plot_base64 = plot_to_base64(plt.gcf())
    plt.close()
    
    return {
        'status': 'success',
        'plot': plot_base64,
        'message': 'Prognostic analysis completed successfully'
    }

if __name__ == '__main__':
    app.run(debug=True) 