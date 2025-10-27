# app.py

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2  # OpenCV
import os
import gdown

# --- 1. DEFINE YOUR PYTORCH MODEL ARCHITECTURE ---
CLASS_NAMES = ['Playing Guitar', 'Jogging', 'Typing on Keyboard', 'Waving Hand', 'Drinking Water']
class MyVideoModel(nn.Module):
    def __init__(self):
        super(MyVideoModel, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, len(CLASS_NAMES)) 
    def forward(self, x):
        x = self.relu(self.conv3d(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- 2. SETUP & MODEL LOADING ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- NEW: Download Model from Google Drive ---
MODEL_PATH = 'your_model.pth'
MODEL_FILE_ID = '1ctIehUfLsoH1yk5KeMmU-b7-Xa79BMLz' # 👈 PASTE YOUR FILE ID HERE

if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found. Downloading...")
    try:
        gdown.download(id=MODEL_FILE_ID, output=MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        # This will stop the app if the download fails
        raise SystemExit()
else:
    print(f"✅ Model file '{MODEL_PATH}' already exists.")

# Now, load the model as usual
try:
    model = MyVideoModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ PyTorch model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise SystemExit()

# --- 3. PREPROCESSING FUNCTION ---
def preprocess_video(video_path):
    NUM_FRAMES_TO_SAMPLE = 16
    FRAME_HEIGHT = 112
    FRAME_WIDTH = 112
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_SAMPLE, dtype=int)
    else:
        raise ValueError("Could not read frames from video file.")
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0: frame = frames[-1]
            else: frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frames.append(frame)
    cap.release()
    processed_frames = [transform(frame) for frame in frames]
    video_tensor = torch.stack(processed_frames, dim=1) 
    video_tensor = video_tensor.unsqueeze(0).to(device)
    print(f"✅ Video processed successfully. Tensor shape: {video_tensor.shape}")
    return video_tensor

# --- 4. FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']

    # <-- NEW: AVI format validation -->
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not video_file.filename.lower().endswith('.avi'):
        return jsonify({'error': 'Invalid file type. Please upload an .avi file.'}), 400

    # Save with the correct extension for clarity
    temp_video_path = "temp_video.avi" 
    video_file.save(temp_video_path)

    try:
        processed_video = preprocess_video(temp_video_path)
        with torch.no_grad():
            outputs = model(processed_video)
            _, predicted_index_tensor = torch.max(outputs, 1)
            predicted_index = predicted_index_tensor.item()
            predicted_class = CLASS_NAMES[predicted_index]
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# if __name__ == '__main__':
#     # Remember to run on a different port if 5000 is in use
#     app.run(debug=True, host='0.0.0.0', port=5001)