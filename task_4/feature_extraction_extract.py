import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

data_dir = '../dataset'
annotations_file = f'{data_dir}/development_scene_annotations.csv'
annotations = pd.read_csv(annotations_file)
output_dir = f'{data_dir}/scenes/extracted_features'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Define the feature extraction function
def extract_features(y, sr):
    features = []
    feature_names = []

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y).flatten()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()

    # Add spectral features
    features.append(spectral_centroid)
    feature_names.append('centroid_0')
    features.append(spectral_bandwidth)
    feature_names.append('bandwidth_0')
    for i, contrast in enumerate(spectral_contrast):
        features.append(contrast)
        feature_names.append(f'contrast_{i}')
    features.append(spectral_flatness)
    feature_names.append('flatness_0')
    features.append(spectral_rolloff)
    feature_names.append('rolloff_0')

    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr).flatten()
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).flatten()

    # Add chroma features
    features.append(chroma_stft)
    feature_names.append('chroma_stft_0')
    features.append(chroma_cqt)
    feature_names.append('chroma_cqt_0')
    features.append(chroma_cens)
    feature_names.append('chroma_cens_0')

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr).flatten()
    features.append(mel_spec)
    feature_names.append('melspect_0')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).flatten()
    for i in range(40):
        features.append(mfcc[i])
        feature_names.append(f'mfcc_{i}')

    # RMS energy
    rms = librosa.feature.rms(y=y).flatten()
    features.append(rms)
    feature_names.append('rms_0')

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    features.append(zcr)
    feature_names.append('zcr_0')

    # Harmonics and Perceived Pitch
    harmony = librosa.effects.harmonic(y).flatten()
    features.append(harmony)
    feature_names.append('harmonic_0')

    # Stack all features into a single array
    features = np.hstack(features)

    return features, feature_names


# Function to save features and feature names
def save_features(annotations, output_dir):
    feature_names_set = None
    max_len = 0

    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        audio_file = os.path.join(f'{data_dir}/scenes/wav', row['filename'] + '.wav')
        y, sr = librosa.load(audio_file, sr=None)
        features, feature_names = extract_features(y, sr)

        if feature_names_set is None:
            feature_names_set = feature_names

        max_len = max(max_len, features.shape[0])

        feature_file = os.path.join(output_dir, row['filename'] + '.npy')
        np.save(feature_file, features)

    return feature_names_set, max_len


print("Extracting features...")
extracted_feature_names, max_len = save_features(annotations, output_dir)
print("Feature extraction complete.")

# Save the feature names
feature_names_file = os.path.join(output_dir, 'idx_to_extracted_feature_names.csv')
pd.DataFrame({'feature_name': extracted_feature_names}).to_csv(feature_names_file, index=False)


# Pad features to the same length
def pad_features(feature_files, max_len):
    padded_features = []
    for f in feature_files:
        features = np.load(os.path.join(output_dir, f))
        if features.shape[0] < max_len:
            features = np.pad(features, (0, max_len - features.shape[0]), mode='constant')
        padded_features.append(features)
    return np.array(padded_features)


# Load and pad extracted features
extracted_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
extracted_features = pad_features(extracted_files, max_len)

# Calculate mean and stddev for the extracted features
extracted_mean = np.mean(extracted_features, axis=0)
extracted_std = np.std(extracted_features, axis=0)

# Save mean and stddev
np.save(os.path.join(output_dir, 'mean.npy'), extracted_mean)
np.save(os.path.join(output_dir, 'std.npy'), extracted_std)
