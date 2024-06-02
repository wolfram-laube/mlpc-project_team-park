import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

data_dir = '../dataset'
preprocessed_data_dir = os.path.join(data_dir, 'preprocessed_audio')

if not os.path.exists(preprocessed_data_dir):
    os.makedirs(preprocessed_data_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the CSV file containing the annotations
annotations = pd.read_csv(f'{data_dir}/development_scene_annotations.csv')

# Add a new category for background noise/unrecognized commands
unrecognized_command_label = 'unrecognized_command'


# Function to check and handle NaNs and Infs
def check_and_handle_nans_infs(array):
    if np.isnan(array).any() or np.isinf(array).any():
        array = np.nan_to_num(array)
        array[np.isinf(array)] = 0
    return array


# Function to augment audio data
def augment_audio(audio, sample_rate):
    augmented_audio = []

    # Original
    augmented_audio.append(audio)

    # Add noise
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise
    augmented_audio.append(audio_noise)

    # Time shift
    shift_range = int(sample_rate * 0.1)  # shift by up to 10% of sample rate
    shift = np.random.randint(-shift_range, shift_range)
    audio_shift = np.roll(audio, shift)
    augmented_audio.append(audio_shift)

    # Change pitch
    pitch_factor = np.random.uniform(-2, 2)
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)
    augmented_audio.append(audio_pitch)

    # Time stretch
    stretch_factor = np.random.uniform(0.8, 1.2)
    audio_stretch = librosa.effects.time_stretch(audio, rate=stretch_factor)
    augmented_audio.append(audio_stretch)

    # Dynamic Range Compression
    audio_drc = librosa.effects.percussive(audio)
    augmented_audio.append(audio_drc)

    # Random Erasing (Zeroing out a part of the audio)
    erase_length = np.random.randint(0, int(len(audio) * 0.1))
    erase_start = np.random.randint(0, len(audio) - erase_length)
    audio_erased = audio.copy()
    audio_erased[erase_start:erase_start + erase_length] = 0
    augmented_audio.append(audio_erased)

    return augmented_audio


# First pass to determine max_length after augmentation
def first_pass(annotations):
    all_lengths = []

    for index, row in annotations.iterrows():
        file_name = f"{data_dir}/scenes/wav/{row['filename']}.wav"
        start_time = row['start']
        end_time = row['end']

        # Load audio file
        audio, sample_rate = librosa.load(file_name, sr=None)

        # Extract the relevant segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = audio[start_sample:end_sample]

        # Normalize the audio segment
        audio_segment = librosa.util.normalize(audio_segment)

        # Apply data augmentation
        augmented_segments = augment_audio(audio_segment, sample_rate)

        for segment in augmented_segments:
            all_lengths.append(len(segment))

    return all_lengths, sample_rate


# Determine the maximum length after augmentation
lengths, sample_rate = first_pass(annotations)
max_length = int(np.percentile(lengths, 95))
logging.info(f"Max length determined statistically: {max_length}")

# Calculate number of augmentations
sample_audio = np.zeros(100)  # Dummy audio segment for augmentation count
M = len(augment_audio(sample_audio, sample_rate))

# Count the number of original segments
N = len(annotations)

# Calculate total number of segments
T = 2 * N * M

logging.info(f"Total number of segments: {T}")


# Second pass to preprocess and pad/crop audio segments
def second_pass(annotations, max_length, sample_rate, max_execution_time=None):
    start_time = time.time()
    scaler = StandardScaler()
    ica = FastICA(n_components=1, whiten='unit-variance')
    file_counter = 0

    logging.info("Starting preprocessing...")

    with tqdm(total=N * M, desc="Processing annotated segments") as pbar1:
        for index, row in annotations.iterrows():
            if max_execution_time and (time.time() - start_time) > max_execution_time:
                logging.info("Max execution time reached. Stopping preprocessing.")
                break

            file_name = f"{data_dir}/scenes/wav/{row['filename']}.wav"
            segment_start_time = row['start']
            segment_end_time = row['end']

            # Load audio file
            audio, sample_rate = librosa.load(file_name, sr=None)

            # Extract the relevant segment
            start_sample = int(segment_start_time * sample_rate)
            end_sample = int(segment_end_time * sample_rate)
            audio_segment = audio[start_sample:end_sample]

            # Normalize the audio segment
            audio_segment = librosa.util.normalize(audio_segment)

            # Apply data augmentation
            augmented_segments = augment_audio(audio_segment, sample_rate)

            for segment in augmented_segments:
                if max_execution_time and (time.time() - start_time) > max_execution_time:
                    logging.info("Max execution time reached. Stopping preprocessing.")
                    break

                # Scale and apply ICA
                segment_scaled = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
                segment_scaled = check_and_handle_nans_infs(segment_scaled)
                segment_ica = ica.fit_transform(segment_scaled.reshape(-1, 1)).flatten()
                segment_ica = check_and_handle_nans_infs(segment_ica)

                # Pad/crop to max_length
                if len(segment_ica) > max_length:
                    segment_ica = segment_ica[:max_length]
                else:
                    segment_ica = np.pad(segment_ica, (0, max_length - len(segment_ica)), 'constant')

                # Save the processed segment as a WAV file with label encoded in the filename
                file_counter += 1
                segment_filename = os.path.join(preprocessed_data_dir, f'segment_{file_counter}_{row["command"]}.wav')
                sf.write(segment_filename, segment_ica, sample_rate)

                pbar1.update(1)

    with tqdm(total=N * M, desc="Processing unrecognized segments") as pbar2:
        for index, row in annotations.iterrows():
            if max_execution_time and (time.time() - start_time) > max_execution_time:
                logging.info("Max execution time reached. Stopping preprocessing.")
                break

            file_name = f"{data_dir}/scenes/wav/{row['filename']}.wav"
            audio, sample_rate = librosa.load(file_name, sr=None)

            # Randomly select segments from the audio that do not overlap with commands
            total_duration = librosa.get_duration(y=audio, sr=sample_rate)
            segment_duration = max_length / sample_rate

            for _ in range(file_counter // len(np.unique([label for label in annotations['command']] + [
                unrecognized_command_label]))):  # Ensure a balanced dataset
                if max_execution_time and (time.time() - start_time) > max_execution_time:
                    logging.info("Max execution time reached. Stopping preprocessing.")
                    break

                segment_start_time = np.random.uniform(0, total_duration - segment_duration)
                start_sample = int(segment_start_time * sample_rate)
                end_sample = start_sample + max_length

                if end_sample > len(audio):
                    continue

                audio_segment = audio[start_sample:end_sample]

                # Normalize and augment the audio segment
                audio_segment = librosa.util.normalize(audio_segment)
                augmented_segments = augment_audio(audio_segment, sample_rate)

                for segment in augmented_segments:
                    if max_execution_time and (time.time() - start_time) > max_execution_time:
                        logging.info("Max execution time reached. Stopping preprocessing.")
                        break

                    # Scale and apply ICA
                    segment_scaled = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
                    segment_scaled = check_and_handle_nans_infs(segment_scaled)
                    segment_ica = ica.fit_transform(segment_scaled.reshape(-1, 1)).flatten()
                    segment_ica = check_and_handle_nans_infs(segment_ica)

                    # Pad/crop to max_length
                    if len(segment_ica) > max_length:
                        segment_ica = segment_ica[:max_length]
                    else:
                        segment_ica = np.pad(segment_ica, (0, max_length - len(segment_ica)), 'constant')

                    # Save the processed segment as a WAV file with label encoded in the filename
                    file_counter += 1
                    segment_filename = os.path.join(preprocessed_data_dir, f'segment_{file_counter}_{unrecognized_command_label}.wav')
                    sf.write(segment_filename, segment_ica, sample_rate)

                    pbar2.update(1)

# Load and preprocess the audio data with padding/cropping
max_execution_time = 120  # Set maximum execution time in seconds (e.g., 10 minutes)
second_pass(annotations, max_length, sample_rate, max_execution_time)

logging.info("Preprocessing completed.")
