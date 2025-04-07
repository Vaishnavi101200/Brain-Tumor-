from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from data_preprocessing import DataPreprocessor
from gan_model import GAN
from prognostic_model import PrognosticModel, PrognosticTrainer
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = GAN(device=device)
prognostic_model = PrognosticModel()
prognostic_trainer = PrognosticTrainer(prognostic_model, device=device)
data_preprocessor = DataPreprocessor()

# Load saved models
gan.load_models('models/gan_model.pth')
prognostic_trainer.load_model('models/prognostic_model.pth')

ALLOWED_EXTENSIONS = {'nii', 'h5'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the file based on its extension
            if filename.endswith('.nii'):
                # Process NIfTI file for tumor detection
                data = data_preprocessor.load_nii_file(file_path)
                data = data_preprocessor.normalize_nii(data)
                data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                
                # Generate tumor mask
                tumor_mask = gan.predict(data)
                tumor_mask = tumor_mask.squeeze().cpu().numpy()
                
                # Calculate tumor volume
                tumor_volume = np.sum(tumor_mask > 0.5)
                
                return jsonify({
                    'status': 'success',
                    'tumor_detected': True if tumor_volume > 0 else False,
                    'tumor_volume': float(tumor_volume),
                    'mask': tumor_mask.tolist()
                })
            
            elif filename.endswith('.h5'):
                # Process H5 file for prognostic analysis
                data = data_preprocessor.load_h5_file(file_path)
                data = data_preprocessor.preprocess_h5(data)
                data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                
                # Get prognosis prediction
                prognosis = prognostic_trainer.predict(data)
                prognosis_classes = ['Good', 'Moderate', 'Poor']
                
                return jsonify({
                    'status': 'success',
                    'prognosis': prognosis_classes[prognosis.item()],
                    'confidence': float(torch.max(prognostic_model(data), 1)[0].item())
                })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000) 