# prediction_script.py
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import logging
import random
import json
from utils import extract_features, pad_features, prepare_segment
from models import KolmogorovArnoldNetwork

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime=s - %(levelname)s - %(message)s')

data_dir = '../dataset'
audio_dir = f'{data_dir}/scenes/wav'
meta_dir = f'{data_dir}/meta'
kan_model_path = 'best_command_kan_model.keras'
output_file = os.path.join(data_dir, 'predictions/predictions.csv')
command_mapping_path = os.path.join(meta_dir, 'command_mapping.json')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load mean and std
mean = np.load(os.path.join(meta_dir, 'mean.npy'))
std = np.load(os.path.join(meta_dir, 'std.npy'))

# Load the model
logging.info('Loading KAN model...')
kan_model = load_model(kan_model_path, custom_objects={'KolmogorovArnoldNetwork': KolmogorovArnoldNetwork})
logging.info('KAN model loaded.')

# Load command mapping
with open(command_mapping_path, 'r') as f:
    command_mapping = json.load(f)

inverse_command_mapping = {v: k for k, v in command_mapping.items()}

# Function to detect commands in an audio file
def detect_command_pairs(audio, sample_rate, model, max_len, window_size=1.0, step_size=0.5, threshold=0.5):
    audio_len = len(audio) / sample_rate
    command_segments = []

    for start in np.arange(0, audio_len, step_size):
        end = min(start + window_size, audio_len)
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            continue

        segment_features = prepare_segment(segment, sample_rate, mean, std, max_len)
        prediction = model.predict(segment_features)
        predicted_label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        if confidence > threshold:
            command_segments.append((start, end, inverse_command_mapping[predicted_label], confidence))

    return command_segments

# Load annotations
annotations = pd.read_csv(os.path.join(meta_dir, 'development_scene_annotations.csv'))

# Randomly select ten files
selected_annotations = annotations.sample(n=10, random_state=42)

# Find maximum feature length from the selected files
max_feature_len = 0
for index, row in tqdm(selected_annotations.iterrows(), total=selected_annotations.shape[0]):
    file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
    y, sr = librosa.load(file_path, sr=None)
    features, _ = extract_features(y, sr)
    for feature in features:
        if isinstance(feature, np.ndarray):
            if feature.ndim > 1:
                max_feature_len = max(max_feature_len, feature.shape[1])
            else:
                max_feature_len = max(max_feature_len, feature.shape[0])

# List to store all predictions
all_predictions = []

# Predict commands for each selected audio file
for index, row in tqdm(selected_annotations.iterrows(), total=selected_annotations.shape[0]):
    file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
    y, sr = librosa.load(file_path, sr=None)
    predictions = detect_command_pairs(y, sr, kan_model, max_feature_len)
    for pred in predictions:
        all_predictions.append([row['filename']] + list(pred))

# Save all predictions to a single CSV file
df = pd.DataFrame(all_predictions, columns=['filename', 'start', 'end', 'command', 'confidence'])
df.to_csv(output_file, index=False)

logging.info('Prediction complete.')
