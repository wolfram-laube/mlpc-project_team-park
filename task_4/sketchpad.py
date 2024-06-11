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
audio_dir = f'{data_dir}/scenes/preprocessed_segments'
output_dir = f'{data_dir}/scenes/extracted_features'
meta_dir = f'{data_dir}/meta'
annotations_file = f'{data_dir}/development_scene_annotations.csv'
non_command_dir = './unknown_commands'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# Load annotations
logging.info('Loading annotations...')
annotations = pd.read_csv(annotations_file)
logging.info('Annotations loaded.')

# Check if non_command_dir exists
if not os.path.exists(non_command_dir):
    logging.error(f'Non-command directory {non_command_dir} does not exist.')
    raise FileNotFoundError(f'Non-command directory {non_command_dir} does not exist.')

# Use full dataset or a balanced random sample for debugging
debug_mode = True  # Set to False to use the full dataset
if debug_mode:
    known_sample_size = 5
    unknown_sample_size = 5

    known_commands = annotations.sample(n=known_sample_size, random_state=42)
    non_command_files = [f for f in os.listdir(non_command_dir) if f.endswith('.wav')]
    unknown_commands = pd.DataFrame({
        'filename': [f.replace('.wav', '') for f in
                     np.random.choice(non_command_files, unknown_sample_size, replace=False)],
        'command': ['unknown_command'] * unknown_sample_size
    })
    annotations = pd.concat([known_commands, unknown_commands], ignore_index=True)
    logging.info(
        f'Using a balanced random sample of {known_sample_size} known commands and {unknown_sample_size} unknown commands for debugging.')

# Extract and save features
def process_and_save_features(annotations, audio_dir, non_command_dir, output_dir, max_timeframes=None, ica_enabled=False):
    feature_names = []
    max_feature_lens = []

    # First pass to find the maximum length of each feature type
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        if row['command'] != 'unknown_command':
            file_path = os.path.join(audio_dir, f"{row['filename']}_{row['command'].replace(' ', '_')}_{row['start']}_{row['end']}.wav")
        else:
            file_path = os.path.join(non_command_dir, f"{row['filename']}.wav")

        y, sr = librosa.load(file_path, sr=None)
        features, _ = extract_features(y, sr, max_len=max_timeframes, ica_enabled=ica_enabled)
        if not max_feature_lens:
            max_feature_lens = [0] * len(features[0])
        for i, feature in enumerate(features[0]):
            max_feature_lens[i] = max(max_feature_lens[i], feature.shape[0])

    # If max_timeframes is provided, override the calculated lengths
    if max_timeframes:
        max_feature_lens = [max_timeframes] * len(max_feature_lens)

    # Second pass to extract features and pad them
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        if row['command'] != 'unknown_command':
            file_path = os.path.join(audio_dir, f"{row['filename']}_{row['command'].replace(' ', '_')}_{row['start']}_{row['end']}.wav")
        else:
            file_path = os.path.join(non_command_dir, f"{row['filename']}.wav")

        y, sr = librosa.load(file_path, sr=None)
        features, feature_names = extract_features(y, sr, max_len=max_timeframes, ica_enabled=ica_enabled)
        padded_features = pad_features(features, max_feature_lens)
        logging.info(f'Extracted features for {os.path.basename(file_path)}: shape {padded_features.shape}')
        np.save(os.path.join(output_dir, row['filename']), padded_features)

    return feature_names, max_feature_lens

# Define the maximum number of timeframes (optional)
max_timeframes = None
ica_enabled = False  # Set to True to enable ICA cleaning

logging.info('Extracting features...')
feature_names, max_feature_lens = process_and_save_features(annotations, audio_dir, non_command_dir, output_dir, max_timeframes, ica_enabled)

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
