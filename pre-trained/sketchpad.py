# Install necessary libraries if not already installed
#!pip install transformers librosa torch datasets noisereduce evaluate accelerate jiwer

import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import noisereduce as nr
import random
import json

# Set environment variable for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Define directories
data_dir = '../dataset'
scenes_dir = f'{data_dir}/scenes/wav'
words_dir = f'{data_dir}/words'

# Function to load audio files
def load_audio_files(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            y, sr = librosa.load(filepath, sr=16000)
            y = y.astype(np.float32)  # Ensure all audio data is of type float32
            audio_files.append({"path": filepath, "audio": y, "sr": sr})
    return audio_files

# Load datasets
scenes_data = load_audio_files(scenes_dir)
words_data = load_audio_files(words_dir)

# Function to add noise
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio)).astype(np.float32)
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

# Function to augment data
def augment_data(audio_files):
    augmented_files = []
    for item in audio_files:
        augmented_audio = add_noise(item["audio"])
        augmented_files.append({"path": item["path"], "audio": augmented_audio, "sr": item["sr"]})
    return augmented_files

# Augment datasets
augmented_scenes_data = augment_data(scenes_data)
augmented_words_data = augment_data(words_data)

# Function to create a dataset
def create_dataset(audio_files):
    data = {"path": [], "audio": [], "text": []}
    for item in audio_files:
        data["path"].append(item["path"])
        data["audio"].append(item["audio"].tolist())  # Convert numpy array to list
        text = os.path.basename(item["path"]).split('.')[0]
        data["text"].append(text)
    return Dataset.from_dict(data)

# Create datasets
scenes_dataset = create_dataset(scenes_data + augmented_scenes_data)
words_dataset = create_dataset(words_data + augmented_words_data)

dataset = DatasetDict({"train": scenes_dataset, "test": words_dataset})

# Function to convert audio to speech
def audio_to_speech(batch):
    batch["speech"] = batch["audio"]
    return batch

# Map audio to speech
dataset = dataset.map(audio_to_speech, remove_columns=["audio"])

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")

# Preprocess the dataset
def preprocess(batch):
    input_values = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding=True).input_values
    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids
    batch["input_values"] = input_values[0].numpy().tolist()  # Convert tensor to list
    batch["labels"] = labels
    return batch

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=["path", "text"])

# Define a custom data collator
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# Training arguments with reduced batch size
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned-noisy",
    per_device_train_batch_size=2,  # Reduced batch size
    evaluation_strategy="steps",
    num_train_epochs=3,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
)

# Define the compute metrics function
import evaluate

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Define the trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

# Train the model with cache clearing
trainer.train()
torch.cuda.empty_cache()  # Clear cache after training

# Evaluate the model
results = trainer.evaluate()
print("Test set WER:", results["eval_wer"])
