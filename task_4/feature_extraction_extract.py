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
preprocessed_segments_dir = f'{data_dir}/scenes/preprocessed_segments'
output_dir = f'{data_dir}/scenes/extracted_features'
meta_dir = f'{data_dir}/meta'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# Extract and save features
def process_and_save_features(segment_dir, output_dir, max_timeframes=None, debug_mode=False):
    feature_names = []
    max_feature_lens = []

    # Get list of files in the segment directory
    files = [f for f in os.listdir(segment_dir) if f.endswith('.wav')]

    # In debug mode, use a smaller sample of files
    if debug_mode:
        known_commands = [f for f in files if "_speech_true_" in f or "_speech_false_" in f][:5]
        unknown_commands = [f for f in files if "speech_" in f and "_speech_true_" not in f and "_speech_false_" not in f][:5]
        noise_commands = [f for f in files if "noise_" in f][:5]
        files = known_commands + unknown_commands + noise_commands
        logging.info(f'Using a debug sample of {len(files)} files for feature extraction.')

    # First pass to find the maximum length of each feature type
    for file in tqdm(files):
        file_path = os.path.join(segment_dir, file)
        y, sr = librosa.load(file_path, sr=None)
        features, _ = extract_features(y, sr, max_len=max_timeframes)
        if not max_feature_lens:
            max_feature_lens = [0] * len(features[0])
        for i, feature in enumerate(features[0]):
            max_feature_lens[i] = max(max_feature_lens[i], feature.shape[0])

    # If max_timeframes is provided, override the calculated lengths
    if max_timeframes:
        max_feature_lens = [max_timeframes] * len(max_feature_lens)

    # Second pass to extract features and pad them
    for file in tqdm(files):
        file_path = os.path.join(segment_dir, file)
        y, sr = librosa.load(file_path, sr=None)
        features, feature_names = extract_features(y, sr, max_len=max_timeframes)
        padded_features = pad_features(features, max_feature_lens)
        logging.info(f'Extracted features for {file}: shape {padded_features.shape}')
        np.save(os.path.join(output_dir, file.replace('.wav', '')), padded_features)

    return feature_names, max_feature_lens

# Define the maximum number of timeframes (optional)
max_timeframes = None

# Toggle debug mode
debug_mode = False

logging.info('Extracting features...')
feature_names, max_feature_lens = process_and_save_features(preprocessed_segments_dir, output_dir, max_timeframes, debug_mode)

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
mean = np.mean(stacked_features, axis=0)
std = np.std(stacked_features, axis=0)

# Ensure no division by zero
std[std == 0] = 1

logging.info(f'Mean shape: {mean.shape}, Std shape: {std.shape}')
np.save(os.path.join(meta_dir, 'mean.npy'), mean)
np.save(os.path.join(meta_dir, 'std.npy'), std)
logging.info('Feature extraction and normalization complete.')
