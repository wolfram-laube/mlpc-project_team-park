import os
import numpy as np
import librosa
import logging
import pandas as pd
from tqdm import tqdm

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

# Function to extract features
def extract_features(y, sr):
    features = []
    feature_names = []

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    for i in range(mel_spec.shape[0]):
        features.append(mel_spec[i])
        feature_names.append(f'melspect_{i}')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    for i in range(mfcc.shape[0]):
        features.append(mfcc[i])
        feature_names.append(f'mfcc_{i}')

    # Delta MFCC
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(mfcc_delta.shape[0]):
        features.append(mfcc_delta[i])
        feature_names.append(f'mfcc_d_{i}')

    # Delta-Delta MFCC
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(mfcc_delta2.shape[0]):
        features.append(mfcc_delta2[i])
        feature_names.append(f'mfcc_d2_{i}')

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    for i in range(bandwidth.shape[0]):
        features.append(bandwidth[i])
        feature_names.append(f'bandwidth_{i}')

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    for i in range(centroid.shape[0]):
        features.append(centroid[i])
        feature_names.append(f'centroid_{i}')

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        features.append(contrast[i])
        feature_names.append(f'contrast_{i}')

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    for i in range(flatness.shape[0]):
        features.append(flatness[i])
        feature_names.append(f'flatness_{i}')

    # Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(flux)
    feature_names.append('flux_0')

    # Root mean square energy
    rms = librosa.feature.rms(y=y)
    for i in range(rms.shape[0]):
        features.append(rms[i])
        feature_names.append(f'energy_{i}')

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    for i in range(zcr.shape[0]):
        features.append(zcr[i])
        feature_names.append(f'zcr_{i}')

    # Power
    power = librosa.feature.rms(y=y)
    for i in range(power.shape[0]):
        features.append(power[i])
        feature_names.append(f'power_{i}')

    # Yin (pitch detection)
    yin = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features.append(yin)
    feature_names.append('yin_0')

    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma_stft.shape[0]):
        features.append(chroma_stft[i])
        feature_names.append(f'chroma_stft_{i}')

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(chroma_cqt.shape[0]):
        features.append(chroma_cqt[i])
        feature_names.append(f'chroma_cqt_{i}')

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(chroma_cens.shape[0]):
        features.append(chroma_cens[i])
        feature_names.append(f'chroma_cens_{i}')

    # Harmonic
    harmonic = librosa.effects.harmonic(y)
    features.append(harmonic)
    feature_names.append('harmonic_0')

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    for i in range(rolloff.shape[0]):
        features.append(rolloff[i])
        feature_names.append(f'rolloff_{i}')

    return np.array(features), feature_names

# Extract and save features
def process_and_save_features(annotations, audio_dir, output_dir):
    feature_names = []
    max_feature_len = 0

    # First pass to find the maximum length of features
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
        y, sr = librosa.load(file_path, sr=None)
        features, _ = extract_features(y, sr)
        for feature in features:
            if isinstance(feature, np.ndarray):
                max_feature_len = max(max_feature_len, len(feature))

    # Second pass to extract features and pad them
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
        y, sr = librosa.load(file_path, sr=None)
        features, feature_names = extract_features(y, sr)

        # Pad each feature to the maximum length
        padded_features = []
        for feature in features:
            if isinstance(feature, np.ndarray):
                if len(feature) < max_feature_len:
                    feature = np.pad(feature, (0, max_feature_len - len(feature)), mode='constant')
            else:
                feature = np.array([feature] * max_feature_len)
            padded_features.append(feature)

        padded_features = np.array(padded_features)
        logging.info(f'Extracted features for {row["filename"]}: shape {padded_features.shape}')
        np.save(os.path.join(output_dir, row['filename']), padded_features)

    return feature_names

logging.info('Extracting features...')
feature_names = process_and_save_features(annotations, audio_dir, output_dir)

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

extracted_features = np.concatenate(extracted_features, axis=0)

mean = np.mean(extracted_features, axis=(0, 2), keepdims=True)
std = np.std(extracted_features, axis=(0, 2), keepdims=True)

# Ensure no division by zero
std[std == 0] = 1

logging.info(f'Mean shape: {mean.shape}, Std shape: {std.shape}')
np.save(os.path.join(meta_dir, 'mean.npy'), mean)
np.save(os.path.join(meta_dir, 'std.npy'), std)
logging.info('Feature extraction and normalization complete.')
