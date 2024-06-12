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

