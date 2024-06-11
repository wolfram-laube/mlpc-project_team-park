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
from utils import extract_features, pad_features
from models import KolmogorovArnoldNetwork
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
audio_dir = f'{data_dir}/scenes/wav'
meta_dir = f'{data_dir}/meta'
kan_model_path = os.path.join(meta_dir, 'best_kan_model.keras')
output_file = os.path.join(data_dir, 'predictions/predictions.csv')
classes_path = os.path.join(meta_dir, 'classes.npy')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load mean and std
mean = np.load(os.path.join(meta_dir, 'mean.npy'))
std = np.load(os.path.join(meta_dir, 'std.npy'))

# Load the model
logging.info('Loading KAN model...')
kan_model = load_model(kan_model_path, custom_objects={'KolmogorovArnoldNetwork': KolmogorovArnoldNetwork})
logging.info('KAN model loaded.')

# Load class labels
class_labels = np.load(classes_path)
label_encoder = LabelEncoder()
label_encoder.classes_ = class_labels

# Function to prepare a segment for prediction
def prepare_segment(segment, sample_rate, mean, std, max_len):
    features, _ = extract_features(segment, sample_rate, max_len=max_len)
    features = pad_features(features, [max_len] * features.shape[1])
    features = (features - mean) / std  # Normalize features
    return features

# Function to detect commands in an audio file
def detect_command_pairs(audio, sample_rate, model, max_len, window_size=1.0, step_size=0.5, threshold=0.5):
    audio_len = len(audio) / sample_rate
    command_segments = []

    for start in np.arange(0, audio_len, step_size):
        end = min(start + window_size, audio_len)
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = audio[start_sample:end_sample]

        if len(segment) < 2048: # minimum reasonable segment length for librosa function calls
            segment = np.pad(segment, (0, 2048 - len(segment)), mode="constant")

        segment_features = prepare_segment(segment, sample_rate, mean, std, max_len)
        prediction = model.predict(segment_features)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        if confidence > threshold:
            predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
            command_segments.append((start, end, predicted_label, confidence))

    return command_segments

# Load audio files and select files for prediction
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
use_random_sample = True  # Set this to False to use the full dataset

if use_random_sample:
    selected_files = random.sample(audio_files, 50)
    logging.info('Using a random sample of 50 files for prediction.')
else:
    selected_files = audio_files
    logging.info('Using the full dataset for prediction.')

# List to store all predictions
all_predictions = []

# Find the maximum length used in training (for padding/cropping)
max_len = std.shape[-1]

# Predict commands for each selected audio file
for audio_file in tqdm(selected_files):
    file_path = os.path.join(audio_dir, audio_file)
    y, sr = librosa.load(file_path, sr=None)
    predictions = detect_command_pairs(y, sr, kan_model, max_len, window_size=2.5, step_size=0.2)
    for pred in predictions:
        all_predictions.append([audio_file] + list(pred))

# Save all predictions to a single CSV file
df = pd.DataFrame(all_predictions, columns=['filename', 'start', 'end', 'command', 'confidence'])
df.to_csv(output_file, index=False)

logging.info('Prediction complete.')

# Generate classification report and confusion matrix
true_commands = []
predicted_commands = []

# Extract true commands from filenames and match with predictions
for _, row in df.iterrows():
    filename_parts = row['filename'].split('_')
    if 'speech' in filename_parts or 'noise' in filename_parts:
        true_command = filename_parts[0]
    else:
        true_command = filename_parts[-3]  # Assumes the command is the third last part
    true_commands.append(true_command)
    predicted_commands.append(row['command'])

# Create classification report
report = classification_report(true_commands, predicted_commands, output_dict=True)
print(classification_report(true_commands, predicted_commands))

# Create confusion matrix
conf_matrix = confusion_matrix(true_commands, predicted_commands)
conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save classification report to JSON
report_file = os.path.join(data_dir, 'predictions/classification_report.json')
with open(report_file, 'w') as f:
    json.dump(report, f)

logging.info('Classification report and heatmap generated.')
