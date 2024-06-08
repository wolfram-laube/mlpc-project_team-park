# feature_extraction_extract.py
import os
import numpy as np
import librosa
import logging
import pandas as pd
from tqdm import tqdm
from utils import extract_features, pad_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
audio_dir = f'{data_dir}/scenes/wav'
output_dir = f'{data_dir}/scenes/extracted_features'
meta_dir = f'{data_dir}/meta'
annotations_file = f'{data_dir}/development_scene_annotations.csv'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# Load annotations
logging.info('Loading annotations...')
annotations = pd.read_csv(annotations_file)
logging.info('Annotations loaded.')

# Extract and save features
def process_and_save_features(annotations, audio_dir, output_dir, max_timeframes=None):
    feature_names = []
    max_feature_lens = []

    # First pass to find the maximum length of each feature type
    for index, row in tqdm(annotations.sample(5).iterrows(), total=5):  # Sample 5 files randomly
        file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
        y, sr = librosa.load(file_path, sr=None)
        features, _ = extract_features(y, sr, max_len=max_timeframes)
        if not max_feature_lens:
            max_feature_lens = [0] * len(features)
        for i, feature in enumerate(features):
            if feature.ndim == 1:
                max_feature_lens[i] = max(max_feature_lens[i], len(feature))
            else:
                max_feature_lens[i] = max(max_feature_lens[i], feature.shape[1])

    # If max_timeframes is provided, override the calculated lengths
    if max_timeframes:
        max_feature_lens = [max_timeframes] * len(max_feature_lens)

    # Second pass to extract features and pad them
    for index, row in tqdm(annotations.sample(5).iterrows(), total=5):  # Sample 5 files randomly
        file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
        y, sr = librosa.load(file_path, sr=None)
        features, feature_names = extract_features(y, sr, max_len=max_timeframes)
        padded_features = pad_features(features, max_feature_lens)
        padded_features = np.squeeze(padded_features, axis=1)  # Remove the singleton dimension
        logging.info(f'Extracted features for {row["filename"]}: shape {padded_features.shape}')
        np.save(os.path.join(output_dir, row['filename']), padded_features)

    return feature_names, max_feature_lens

# Define the maximum number of timeframes (optional)
max_timeframes = 500

logging.info('Extracting features...')
feature_names, max_feature_lens = process_and_save_features(annotations, audio_dir, output_dir, max_timeframes)

# Save the feature names
feature_names_file = os.path.join(meta_dir, 'idx_to_extracted_feature_names.csv')
pd.DataFrame({'index': range(len(feature_names)), 'feature_name': feature_names}).to_csv(feature_names_file, index=False)

# Calculate mean and std
logging.info('Calculating mean and std...')
extracted_features = []
for file in os.listdir(output_dir):
    if file.endswith('.npy'):
        features = np.load(os.path.join(output_dir, file))
        extracted_features.append(features)

# Stack features along a new dimension
stacked_features = np.stack(extracted_features, axis=0)

# Calculate mean and std across the new dimension
mean = np.mean(stacked_features, axis=(0), keepdims=True)
std = np.std(stacked_features, axis=(0), keepdims=True)

# Ensure no division by zero
std[std == 0] = 1

logging.info(f'Mean shape: {mean.shape}, Std shape: {std.shape}')
np.save(os.path.join(meta_dir, 'mean.npy'), mean)
np.save(os.path.join(meta_dir, 'std.npy'), std)
logging.info('Feature extraction and normalization complete.')
