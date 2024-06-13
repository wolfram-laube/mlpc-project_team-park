#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/wolfram-laube/mlpc-project_team-park/blob/wl/pre-trained/fastlane.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # All-in-one Pre-trained Word Tokenizer

# In[20]:


data_dir = '/content/dataset'
data_dir = '../dataset'


# In[ ]:


# Install necessary libraries if not already installed
get_ipython().system('pip install transformers librosa torch datasets noisereduce evaluate jiwer pandas')



# ## Preproccess

# ### Load fresh data

# In[ ]:


import os
import sys
import shutil

# Check if the environment is Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # If in Google Colab
    from google.colab import drive
    import gdown

    # Option 1: Download the file by its public link and expand it to the Colab runtime
    import urllib.request
    import zipfile

    scnwavzip_file_id = '1oI1EsH1krrEPbH9MSZRzLHu-_4p6-njR' # https://drive.google.com/file/d/1oI1EsH1krrEPbH9MSZRzLHu-_4p6-njR/view?usp=sharing
    scnnpyzip_file_id = '1oKgurvIgT93RGkxvxq8AA423VKlEVT7O' # https://drive.google.com/file/d/1oKgurvIgT93RGkxvxq8AA423VKlEVT7O/view?usp=sharing
    wrdwavzip_file_id = '1o1yBqdtqH3tjOHN4GKISJHlY2Qyu_ouX' # https://drive.google.com/file/d/1o1yBqdtqH3tjOHN4GKISJHlY2Qyu_ouX/view?usp=sharing
    wrdnpyzip_file_id = '1o2fj6QAM00zg8YMxsHwcNa2lkIXLXDYs' # https://drive.google.com/file/d/1o2fj6QAM00zg8YMxsHwcNa2lkIXLXDYs/view?usp=sharing
    annotation_file_id = '1xLxget7c5nCkwYt9Ru2RpYi5rMkk_pl0'  # https://drive.google.com/file/d/1xLxget7c5nCkwYt9Ru2RpYi5rMkk_pl0/view?usp=sharing
    scenes_file_id = '1xLgB7-cCz6nReyQbFJJcJGOUKCCbNhCG'  # https://drive.google.com/file/d/1xLgB7-cCz6nReyQbFJJcJGOUKCCbNhCG/view?usp=sharing

    scnwavzip_url = f'https://drive.google.com/uc?id={scnwavzip_file_id}'
    scnnpyzip_url = f'https://drive.google.com/uc?id={scnnpyzip_file_id}'
    wrdwavzip_url = f'https://drive.google.com/uc?id={wrdwavzip_file_id}'
    wrdnpyzip_url = f'https://drive.google.com/uc?id={wrdnpyzip_file_id}'
    annotation_url = f'https://drive.google.com/uc?id={annotation_file_id}'
    scenes_url = f'https://drive.google.com/uc?id={scenes_file_id}'

    scnwavzip_path = '/content/scenes_data.zip'
    scnnpyzip_path = '/content/scenes_feat.zip'
    wrdwavzip_path = '/content/words_data.zip'
    wrdnpyzip_path = '/content/words_feat.zip'
    data_dir = '/content/dataset'
    scenes_dir = f'{data_dir}/scenes'
    words_dir = f'{data_dir}/words'
    scenes_wav_dir = f'{scenes_dir}/wav'
    scenes_npy_dir = f'{scenes_dir}/npy'
    words_wav_dir = f'{data_dir}/words'
    words_npy_dir = f'{data_dir}/words'

    # Download the WAVZIP file
    #urllib.request.urlretrieve(wavzip_url, wavzip_path)
    gdown.download(scnwavzip_url, scnwavzip_path, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(scnwavzip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Scenes training data extracted to {data_dir}")

     # Create the 'scenes/wav' folder structure
    os.makedirs(scenes_wav_dir, exist_ok=True)

    # Copy .wav files to 'scenes/wav'
    extracted_scenes_dir = os.path.join(data_dir, 'mlpc24_speech_commands', 'scenes')
    for root, dirs, files in os.walk(extracted_scenes_dir):
        for file in files:
            if file.endswith('.wav'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(scenes_wav_dir, file)
                shutil.copy(src_path, dst_path)

    print(f"Scenes training .wav files moved to {scenes_wav_dir}")

    # Download the SCNNPYZIP file
    gdown.download(scnnpyzip_url, scnnpyzip_path, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(scnnpyzip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Scenes training features extracted to {data_dir}")

     # Create the 'scenes/npy' folder structure
    os.makedirs(scenes_npy_dir, exist_ok=True)

    # Copy .npy files to 'scenes/npy'
    extracted_scenes_dir = os.path.join(data_dir, 'development_scenes')
    for root, dirs, files in os.walk(extracted_scenes_dir):
        for file in files:
            if file.endswith('.npy'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(scenes_npy_dir, file)
                shutil.copy(src_path, dst_path)

    print(f"Scenes training .npy files moved to {scenes_npy_dir}")

    # Download the WRDWAVZIP file
    #urllib.request.urlretrieve(wavzip_url, wavzip_path)
    gdown.download(wrdwavzip_url, wrdwavzip_path, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(wrdwavzip_path, 'r') as zip_ref:
        zip_ref.extractall(words_wav_dir)

    print(f"Words training data extracted to {words_wav_dir}")

    # Download the WRDNPYZIP file
    gdown.download(wrdnpyzip_url, wrdnpyzip_path, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(wrdnpyzip_path, 'r') as zip_ref:
        zip_ref.extractall(words_npy_dir)

    print(f"Words training ,npy files s extracted to {words_npy_dir}")


    # Download the CSV files into the data_dir
    annotation_orig_path = os.path.join(data_dir, 'development_scene_annotations.csv.orig') # Keep a backup copy because it needs fixing
    annotation_path = os.path.join(data_dir, 'development_scene_annotations.csv')
    scenes_path = os.path.join(data_dir, 'development_scenes.csv')

    gdown.download(annotation_url, annotation_orig_path, quiet=False)
    gdown.download(annotation_url, annotation_path, quiet=False)
    gdown.download(scenes_url, scenes_path, quiet=False)

    print(f"CSV files downloaded to {scenes_dir}")

    # Option 2: Mount Google Drive and use the training data
    # Note this really takes some time for preprocessing file by file
    #drive.mount('/content/drive')
    #data_dir = '/content/drive/My Drive/Dropbox/public/mlpc/dataset'

    # Use this option to read from Google Drive instead
    #print(f"Using training data from {data_dir}")
else:
    # If on local machine
    data_dir = '../dataset'
    print(f"Using local training data from {data_dir}")

# Use the data_dir variable as the path to your training data


# ### Determine CPU/GPU

# In[ ]:


# Function to check if GPU is available
#def is_gpu_available():
#    try:
#        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#        return result.returncode == 0
#    except FileNotFoundError:
#        return False

def is_gpu_available():
    try:
        import torch
        is_gpu = torch.cuda.is_available()
        print(f'GPU available: {is_gpu}')
        return is_gpu
    except ImportError as ie:
        print("No GPU support", ie)
        pass

    try:
        import tensorflow as tf
        is_gpu =  tf.config.list_physical_devices('GPU') != []
        print(f'GPU available: {is_gpu}')
        return is_gpu
    except ImportError as ie:
        print("No GPU support", ie)
        pass

    print("No GPU support found")
    return False

is_gpu_available()


# ### Fix erreneous metadata

# #### Before

# In[ ]:


import pandas as pd

# Load the CSV files
scene_annotations_df = pd.read_csv(f'{data_dir}/development_scene_annotations.csv')
scenes_df = pd.read_csv(f'{data_dir}/development_scenes.csv')

# Check the head of the dataframes to understand their structure
print(scene_annotations_df.head())
print(scenes_df.head())

# Check the distribution of labels in the annotations CSV
label_distribution_annotations = scene_annotations_df['command'].value_counts()
print("Label Distribution in development_scene_annotations.csv:")
print(label_distribution_annotations)

# Check the distribution of speaker IDs in the scenes CSV
label_distribution_scenes = scenes_df['speaker_id'].value_counts()
print("Label Distribution in development_scenes.csv:")
print(label_distribution_scenes)


# #### Fix

# In[ ]:


import os
import re
import shutil
import pandas as pd

# Paths to the original and working copy files
original_file_path = f'{data_dir}/development_scene_annotations.csv.orig'
working_copy_path = f'{data_dir}/development_scene_annotations.csv.0'
corrected_file_path = f'{data_dir}/development_scene_annotations.csv'

# Step 1: Create a working copy of the original file
shutil.copy(original_file_path, working_copy_path)

# Step 2: Load the working copy into a DataFrame
df = pd.read_csv(working_copy_path)

# Define the pattern to parse the filename
filename_pattern = re.compile(r'(\d+)_speech_(true|false)_((?:[a-zA-ZäöüÄÖÜß]+_(?:an|aus)_?)+)', re.UNICODE)

# Function to parse filename and extract commands
def parse_filename(filename):
    match = filename_pattern.match(filename)
    if not match:
        return []

    commands_str = match.group(3)
    commands = commands_str.split('_')

    command_list = []
    for i in range(0, len(commands), 2):
        command_list.append(f"{commands[i]} {commands[i+1]}")

    return command_list

# Parse the commands from filenames and add to the DataFrame
df['parsed_commands'] = df['filename'].apply(parse_filename)

# Step 3: Group by filename and sort by start time
grouped = df.groupby('filename').apply(lambda x: x.sort_values(by='start')).reset_index(drop=True)

# Step 4: Assign the correct labels based on the order of commands in the filename
def assign_labels(group):
    commands = group['parsed_commands'].iloc[0]  # get the parsed commands from the first row
    group = group.reset_index(drop=True)
    for i in range(len(group)):
        if i < len(commands):
            group.at[i, 'command'] = commands[i]
        else:
            print(f"Warning: More segments than commands in {group['filename'].iloc[0]}")
    return group

# Apply the label assignment function
corrected_df = grouped.groupby('filename').apply(assign_labels).reset_index(drop=True)

# Drop the temporary column
corrected_df = corrected_df.drop(columns=['parsed_commands'])

# Step 5: Save the corrected DataFrame to a new CSV file
corrected_df.to_csv(corrected_file_path, index=False)

# Verify the saved corrections
print("Label corrections applied and saved successfully.")
print(corrected_df.head())


# #### After

# In[ ]:


import pandas as pd

# Load the CSV files
scene_annotations_df = pd.read_csv(f'{data_dir}/development_scene_annotations.csv')
scenes_df = pd.read_csv(f'{data_dir}/development_scenes.csv')

# Check the head of the dataframes to understand their structure
print(scene_annotations_df.head())
print(scenes_df.head())

# Check the distribution of labels in the annotations CSV
label_distribution_annotations = scene_annotations_df['command'].value_counts()
print("Label Distribution in development_scene_annotations.csv:")
print(label_distribution_annotations)

# Check the distribution of speaker IDs in the scenes CSV
label_distribution_scenes = scenes_df['speaker_id'].value_counts()
print("Label Distribution in development_scenes.csv:")
print(label_distribution_scenes)


# ### Preprocess audio data

# In[ ]:


import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import random
from IPython.display import Audio
from sklearn.decomposition import FastICA

# Function to apply ICA on audio segments
def apply_ica(segment, sr):
    ica = FastICA(n_components=1, whiten='arbitrary-variance')  # Explicitly set whiten parameter
    segment_reshaped = segment.reshape(-1, 1)
    segment_ica = ica.fit_transform(segment_reshaped).flatten()
    return segment_ica

# Function to preprocess segments and optionally save to the filesystem
def preprocess_and_save_segments(scenes_dir, annotations_path, save_dir=None, save_to_filesystem=False, apply_ica_flag=False):
    # Load the annotations
    annotations_df = pd.read_csv(annotations_path)

    # Ensure the save directory exists if saving to filesystem
    if save_to_filesystem and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    preprocessed_segments = []

    for index, row in annotations_df.iterrows():
        filename = row['filename']
        command = row['command']
        start = row['start']
        end = row['end']

        # Load the audio file
        file_path = os.path.join(scenes_dir, f"{filename}.wav")
        y, sr = librosa.load(file_path, sr=None)

        # Extract the segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        # Normalize the segment
        segment = librosa.util.normalize(segment)

        # Apply ICA if the flag is set
        if apply_ica_flag:
            segment = apply_ica(segment, sr)

        # Add the segment to the list
        preprocessed_segments.append((filename, command, segment, sr))

        # Save the segment to the filesystem if required
        if save_to_filesystem and save_dir is not None:
            save_path = os.path.join(save_dir, f"{filename}_{start}_{end}.wav")
            sf.write(save_path, segment, sr)

    return preprocessed_segments

# Function to play a random segment from preprocessed segments
def play_random_segment(preprocessed_segments):
    # Select a random segment
    random_segment = random.choice(preprocessed_segments)

    filename, command, audio_data, sample_rate = random_segment

    # Print the command and play the audio segment
    print(f"Filename: {filename}")
    print(f"Command: {command}")

    return Audio(audio_data, rate=sample_rate)

# Function to play a random segment from the filesystem
def play_random_segment_from_filesystem(save_dir, annotations_path):
    # List all the preprocessed segment files
    segment_files = [f for f in os.listdir(save_dir) if f.endswith('.wav')]

    # Select a random segment file
    random_segment_file = random.choice(segment_files)
    random_segment_path = os.path.join(save_dir, random_segment_file)

    # Extract start and end times from the file name
    filename_parts = random_segment_file.split('_')
    filename = '_'.join(filename_parts[:-2])
    start_time = float(filename_parts[-2])
    end_time = float(filename_parts[-1].replace('.wav', ''))

    # Find the command in the annotations
    annotations_df = pd.read_csv(annotations_path)
    command_row = annotations_df[
        (annotations_df['filename'] == filename) &
        (annotations_df['start'] == start_time) &
        (annotations_df['end'] == end_time)
    ]

    if command_row.empty:
        print(f"No matching annotation found for {random_segment_file}")
        return

    command = command_row.iloc[0]['command']

    # Load the audio segment
    y, sr = librosa.load(random_segment_path, sr=None)

    # Print the command and play the audio segment
    print(f"Filename: {filename}")
    print(f"Command: {command}")

    return Audio(y, rate=sr)

# Example usage
scenes_dir = f'{data_dir}/scenes/wav'
annotations_path = f'{data_dir}/development_scene_annotations.csv'
save_dir = f'{data_dir}/clipped_commands'

# Preprocess segments and save to filesystem with optional ICA
preprocessed_segments = preprocess_and_save_segments(scenes_dir, annotations_path, save_dir, save_to_filesystem=True, apply_ica_flag=True)

# Play a random segment from memory
audio_memory = play_random_segment(preprocessed_segments)
display(audio_memory)

# Play a random segment from filesystem
audio_filesystem = play_random_segment_from_filesystem(save_dir, annotations_path)
display(audio_filesystem)


# ## Main

# ### Libraries

# #### audio_filename_utils.py

# In[ ]:


# audio_parsing_utils.py

import re
import unicodedata
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the pattern to parse clipped command filenames
clipped_command_pattern = re.compile(
    r'(\d+)_speech_(true|false)_((?:[a-zA-ZäöüÄÖÜß]+_(?:an|aus)_?)+)_(\d+\.\d+)_(\d+\.\d+)\.wav', re.UNICODE
)

# Define the pattern to parse full scene filenames
full_scene_pattern = re.compile(
    r'(\d+)_speech_(true|false)_((?:[a-zA-ZäöüÄÖÜß]+_(?:an|aus)_?)+)\.wav', re.UNICODE
)

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

# Function to parse clipped command filenames to extract commands, start time, and end time
def parse_clipped_command_filename(filename):
    logger.debug(f"Attempting to parse filename: {filename}")
    filename = normalize_unicode(filename)
    match = clipped_command_pattern.match(filename)
    if not match:
        logger.error(f"Filename {filename} does not match the expected pattern.")
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

    # Extract command string and timestamps
    commands_str = match.group(3)
    start_time = float(match.group(4))
    end_time = float(match.group(5))

    # Split and format commands
    commands = commands_str.split('_')
    command_list = []
    for i in range(0, len(commands), 2):
        command_list.append(f"{commands[i]} {commands[i+1]}")

    logger.debug(f"Parsed filename {filename}: file_id={match.group(1)}, speech_flag={match.group(2)}, command_list={command_list}, start_time={start_time}, end_time={end_time}")
    return match.group(1), match.group(2), command_list, start_time, end_time

# Function to parse full scene filenames to extract commands
def parse_full_scene_filename(filename):
    logger.debug(f"Attempting to parse filename: {filename}")
    filename = normalize_unicode(filename)
    match = full_scene_pattern.match(filename)
    if not match:
        logger.error(f"Filename {filename} does not match the expected pattern.")
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

    # Extract command string
    file_id = match.group(1)
    speech_flag = match.group(2)
    commands_str = match.group(3)

    # Split and format commands
    commands = commands_str.split('_')
    command_list = []
    for i in range(0, len(commands), 2):
        command_list.append(f"{commands[i]} {commands[i+1]}")

    logger.debug(f"Parsed filename {filename}: file_id={file_id}, speech_flag={speech_flag}, command_list={command_list}")
    return file_id, speech_flag, command_list


# #### audio_loading_utils.py

# In[ ]:


# audio_loading_utils.py

import os
import librosa
import numpy as np
"""
from audio_parsing_utils import (
    parse_clipped_command_filename,
    parse_full_scene_filename
)
"""

# Function to load audio files from the scenes directory
def load_scene_files(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            y, sr = librosa.load(filepath, sr=16000)
            y = y.astype(np.float32)  # Ensure all audio data is of type float32
            file_id, _, commands = parse_full_scene_filename(filename)
            audio_files.append({"path": filepath, "audio": y, "sr": sr, "text": " ".join(commands)})
    return audio_files

# Function to load audio files from the words directory
def load_word_files(directory):
    audio_files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(root, filename)
                y, sr = librosa.load(filepath, sr=16000)
                y = y.astype(np.float32)  # Ensure all audio data is of type float32
                text = os.path.basename(root)  # Extract text from folder name
                audio_files.append({"path": filepath, "audio": y, "sr": sr, "text": text})
    return audio_files

# Function to load audio files from the clipped commands directory
def load_clipped_command_files(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            y, sr = librosa.load(filepath, sr=16000)
            y = y.astype(np.float32)  # Ensure all audio data is of type float32
            _, _, command_list, start_time, end_time = parse_clipped_command_filename(filename)
            command = " ".join(command_list)
            audio_files.append({"path": filepath, "audio": y, "sr": sr, "text": command, "start_time": start_time, "end_time": end_time})
    return audio_files


# #### data_collator.py

# In[ ]:


# data_collator.py

import torch
from transformers import Wav2Vec2Processor

class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": labels},
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding with -100 to ignore them during loss computation
        labels_batch["input_ids"][labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels_batch["input_ids"]

        return batch


# ### Training

# In[ ]:


import os
import torch
import librosa
import logging
import gc
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_metric
"""from audio_loading_utils import (
    load_scene_files,
    load_word_files,
    load_clipped_command_files
)
from data_collator import DataCollatorCTCWithPadding"""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for MPS fallback and high watermark ratio (for Mac)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Detect the device to be used (CPU, CUDA, MPS)
if torch.cuda.is_available():
    device = torch.device("cuda")
    fp16 = True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    fp16 = False  # FP16 is not supported on MPS
else:
    device = torch.device("cpu")
    fp16 = False

logger.info(f"Using device: {device}")

# Set the default tensor type to float32
torch.set_default_dtype(torch.float32)

# Define directories
data_dir = '../dataset'
scenes_dir = f'{data_dir}/scenes/wav'
words_dir = f'{data_dir}/words'
clipped_commands_dir = f'{data_dir}/clipped_commands'
model_dir = f"{data_dir}/meta"
processor_dir = f"{data_dir}/meta"

# Function to load datasets with progress bars
def load_datasets():
    logger.info("Loading datasets...")
    scenes_data = load_scene_files(scenes_dir)
    words_data = load_word_files(words_dir)
    clipped_commands_data = load_clipped_command_files(clipped_commands_dir)
    return scenes_data, words_data, clipped_commands_data

# Function to create a dataset from the audio files
def create_dataset(audio_files):
    data = {"path": [], "audio": [], "text": []}
    for item in tqdm(audio_files, desc="Creating dataset"):
        data["path"].append(item["path"])
        data["audio"].append(item["audio"].tolist())  # Convert numpy array to list
        data["text"].append(item["text"])
    return Dataset.from_dict(data)

# Load datasets with logging and progress bars
scenes_data, words_data, clipped_commands_data = load_datasets()

# Create datasets with progress bars
logger.info("Creating datasets...")
scenes_dataset = create_dataset(scenes_data)
words_dataset = create_dataset(words_data)
clipped_commands_dataset = create_dataset(clipped_commands_data)

# Free memory after dataset creation
del scenes_data, words_data, clipped_commands_data
gc.collect()

# Combine datasets into a DatasetDict
dataset = DatasetDict({"train": scenes_dataset, "test": words_dataset})

# Free memory after combining datasets
del scenes_dataset, words_dataset, clipped_commands_dataset
gc.collect()

# Load the pre-trained model and processor
logger.info("Loading pre-trained model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

# Move model to the detected device
model.to(device)
logger.info("Model moved to device.")

# Preprocess function for dataset
def preprocess(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"], padding="max_length", max_length=128, truncation=True).input_ids
    return batch

# Apply preprocessing with progress bars
logger.info("Applying preprocessing...")
dataset = dataset.map(preprocess, remove_columns=["path", "audio", "text"], num_proc=4, desc="Preprocessing dataset")

# Free memory after preprocessing
gc.collect()

# Define data collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Define metric
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce batch size if necessary
    per_device_eval_batch_size=4,  # Reduce batch size if necessary
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # to disable wandb and other integrations
    gradient_accumulation_steps=2,  # Accumulate gradients
    fp16=fp16,  # Enable FP16 if supported
    bf16=False,  # Ensure BF16 precision is disabled
)

# Initialize Trainer
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
logger.info("Starting training...")
trainer.train()

# Evaluate the new model
logger.info("Evaluating new model...")
new_eval_results = trainer.evaluate()
logger.info(f"New model evaluation results: {new_eval_results}")

# Check if existing model and processor exist
if os.path.exists(model_dir) and os.path.exists(processor_dir):
    logger.info("Loading existing model and processor for comparison...")
    # Load the existing model and processor
    existing_model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    existing_processor = Wav2Vec2Processor.from_pretrained(processor_dir)
    existing_model.to(device)

    # Initialize a new Trainer for the existing model
    existing_trainer = Trainer(
        model=existing_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=existing_processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Evaluate the existing model
    logger.info("Evaluating existing model...")
    existing_eval_results = existing_trainer.evaluate()
    logger.info(f"Existing model evaluation results: {existing_eval_results}")

    # Compare WER (lower is better)
    if new_eval_results["eval_wer"] < existing_eval_results["eval_wer"]:
        logger.info("New model performs better. Saving new model and processor.")
        model.save_pretrained(model_dir)
        processor.save_pretrained(processor_dir)
    else:
        logger.info("Existing model performs better. Keeping existing model and processor.")
else:
    logger.info("No existing model found. Saving new model and processor.")
    model.save_pretrained(model_dir)
    processor.save_pretrained(processor_dir)

# Free memory after saving models
del model, processor
gc.collect()


# In[ ]:


model_dir = f"{data_dir}/meta"
processor_dir = f"{data_dir}/meta"

# Check if existing model and processor exist
if os.path.exists(model_dir) and os.path.exists(processor_dir):
    # Load the existing model and processor
    existing_model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    existing_processor = Wav2Vec2Processor.from_pretrained(processor_dir)
    existing_model.to(device)

    # Initialize a new Trainer for the existing model
    existing_trainer = Trainer(
        model=existing_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=existing_processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Evaluate the existing model
    existing_eval_results = existing_trainer.evaluate()
    print(f"Existing model evaluation results: {existing_eval_results}")

    # Compare WER (lower is better)
    if new_eval_results["eval_wer"] < existing_eval_results["eval_wer"]:
        print("New model performs better. Saving new model and processor.")
        model.save_pretrained(model_dir)
        processor.save_pretrained(processor_dir)
    else:
        print("Existing model performs better. Keeping existing model and processor.")
else:
    print("No existing model found. Saving new model and processor.")
    model.save_pretrained(model_dir)
    processor.save_pretrained(processor_dir)


# In[ ]:


# inference_script.py

import os
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Define the directory where the model and processor are saved
model_dir = "path/to/save/model"
processor_dir = "path/to/save/processor"

# Load the saved model and processor
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
processor = Wav2Vec2Processor.from_pretrained(processor_dir)

# Move model to CPU
device = torch.device("cpu")
model.to(device)

# Function to transcribe an audio file with timestamps
def transcribe(audio_path):
    # Load audio file
    audio_input, sr = librosa.load(audio_path, sr=16000)

    # Preprocess the audio input
    input_values = processor(audio_input, return_tensors="pt", padding="longest").input_values.to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Get the timestamps
    timestamps = []
    input_lengths = input_values.shape[1]
    time_per_input = len(audio_input) / sr / input_lengths
    last_token = None

    for i, token in enumerate(predicted_ids[0]):
        if token != processor.tokenizer.pad_token_id and token != last_token:
            word = processor.tokenizer.decode([token])
            start_time = i * time_per_input
            end_time = (i + 1) * time_per_input
            timestamps.append((word, start_time, end_time))
            last_token = token

    return transcription, timestamps

# Example usage
audio_path = "path/to/audio/file.wav"
transcription, timestamps = transcribe(audio_path)
print(f"Transcription: {transcription}")
for word, start, end in timestamps:
    print(f"Word: {word}, Start time: {start:.2f}, End time: {end:.2f}")

