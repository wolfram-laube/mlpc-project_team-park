import numpy as np
import librosa
import logging
import re
from sklearn.decomposition import FastICA


def extract_info(filename):
    # Regular expression to match the main parts of the filename, including umlauts and special characters
    pattern = r"(\d+)_speech_(true|false)_((?:[A-Za-z0-9äöüÄÖÜß]+_[A-Za-z0-9äöüÄÖÜß]+_)+)(\d+\.\d+)_(\d+\.\d+)\.npy"
    match = re.match(pattern, filename)
    if match:
        file_id = match.group(1)
        speech_status = match.group(2)
        commands_string = match.group(3)[:-1]  # Remove the trailing underscore
        start_time = float(match.group(4))
        end_time = float(match.group(5))

        # Split the command string into pairs
        commands = re.findall(r"[A-Za-z0-9äöüÄÖÜß]+_[A-Za-z0-9äöüÄÖÜß]+", commands_string)

        # The effective command is the last command pair
        effective_command = commands[-1].replace('_', ' ')  # Replace underscores with blanks

        return file_id, speech_status, commands, effective_command, start_time, end_time
    return None


def ica_cleaning(y, sr, n_components=5):
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    ica = FastICA(n_components=n_components, random_state=42)
    ica_result = ica.fit_transform(magnitude.T).T
    reconstructed_stft = ica_result.T * np.exp(1j * np.angle(stft))
    y_clean = librosa.istft(reconstructed_stft)
    return y_clean


def extract_features(y, sr, max_len=None, ica_enabled=False):
    features = []
    feature_names = []

    # ICA cleaning if enabled
    if ica_enabled:
        y = ica_cleaning(y, sr)

    # Determine the n_fft value dynamically
    n_fft = min(len(y), 512)
    n_fft = 2 ** (n_fft - 1).bit_length()  # Ensure n_fft is a power of 2

    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=n_fft)
    for i in range(mel_spec.shape[0]):
        features.append(mel_spec[i])
        feature_names.append(f'melspect_{i}')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32, n_fft=n_fft)
    if mfcc.shape[1] < 3:
        mfcc = np.pad(mfcc, ((0, 0), (0, 3 - mfcc.shape[1])), mode='constant')
    for i in range(mfcc.shape[0]):
        features.append(mfcc[i])
        feature_names.append(f'mfcc_{i}')

    # Delta MFCC
    if mfcc.shape[1] >= 3:
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        for i in range(mfcc_delta.shape[0]):
            features.append(mfcc_delta[i])
            feature_names.append(f'mfcc_d_{i}')

        # Delta-Delta MFCC
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=3)
        for i in range(mfcc_delta2.shape[0]):
            features.append(mfcc_delta2[i])
            feature_names.append(f'mfcc_d2_{i}')
    else:
        # Padding with zeros if the segment is too short
        for _ in range(mfcc.shape[0]):
            features.append(np.zeros(mfcc.shape[1]))
            feature_names.append(f'mfcc_d_{_}')
            features.append(np.zeros(mfcc.shape[1]))
            feature_names.append(f'mfcc_d2_{_}')

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)
    for i in range(bandwidth.shape[0]):
        features.append(bandwidth[i])
        feature_names.append(f'bandwidth_{i}')

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    for i in range(centroid.shape[0]):
        features.append(centroid[i])
        feature_names.append(f'centroid_{i}')

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    for i in range(contrast.shape[0]):
        features.append(contrast[i])
        feature_names.append(f'contrast_{i}')

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft)
    for i in range(flatness.shape[0]):
        features.append(flatness[i])
        feature_names.append(f'flatness_{i}')

    # Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(flux)
    feature_names.append('flux_0')

    # Root mean square (RMS) energy, which measures the power of the signal
    #    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=n_fft//2)
    rms = librosa.feature.rms(y=y, frame_length=n_fft)
    for i in range(rms.shape[0]):
        features.append(rms[i])
        feature_names.append(f'energy_{i}')

    # Zero crossing rate
    #    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=n_fft//2)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft)
    for i in range(zcr.shape[0]):
        features.append(zcr[i])
        feature_names.append(f'zcr_{i}')

    # Yin (pitch detection)
    yin = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features.append(yin)
    feature_names.append('yin_0')

    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
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

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
    for i in range(rolloff.shape[0]):
        features.append(rolloff[i])
        feature_names.append(f'rolloff_{i}')

    # Convert features to a numpy array
    try:
        features = np.array(features)
    except ValueError as ve:
        print(ve)
    try:
        features = np.expand_dims(features, axis=0)  # Add batch dimension
    except ValueError as ve:
        print(ve)

    # If max_len is specified, crop features
    if max_len:
        features = features[:, :, :max_len]

    return features, feature_names


def pad_features(features, max_lens):
    padded_features = []
    for feature, max_len in zip(features[0], max_lens):  # Access the correct axis
        if feature.shape[0] < max_len:
            feature = np.pad(feature, (0, max_len - feature.shape[0]), mode='constant')
        else:
            feature = feature[:max_len]  # Crop to max_len
        padded_features.append(feature)
    return np.expand_dims(np.array(padded_features), axis=0)  # Add batch dimension


def augment_data(features, noise_factor=0.005, stretch_factors=[0.8, 1.2]):
    augmented_features = []
    for feature in features:
        # Add Gaussian noise
        noise = np.random.randn(*feature.shape) * noise_factor
        augmented_feature = feature + noise

        # Time stretching
        for stretch_factor in stretch_factors:
            stretched_feature = librosa.effects.time_stretch(augmented_feature[0], stretch_factor)
            augmented_features.append(np.expand_dims(stretched_feature, axis=0))

        augmented_features.append(augmented_feature)
    return np.array(augmented_features)
