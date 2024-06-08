import numpy as np
import librosa

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

    return np.array(features), feature_names

# Function to pad features
def pad_features(features, max_feature_lens):
    padded_features = []
    for i, feature in enumerate(features):
        max_len = max_feature_lens[i]
        if feature.ndim == 1:
            feature = np.expand_dims(feature, axis=0)
        if feature.shape[1] < max_len:
            feature = np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
        padded_features.append(feature)
    return np.array(padded_features)

# Function to prepare a segment for prediction
def prepare_segment(segment, sample_rate, mean, std):
    features, _ = extract_features(segment, sample_rate)
    features = (features - mean) / std  # Normalize features
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    return features
