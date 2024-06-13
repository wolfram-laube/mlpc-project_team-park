import os
import re
import torch
import librosa
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# Load pre-trained tokenizer and model
def load_model_and_tokenizer(model_name="facebook/wav2vec2-large-960h"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model


# Extract labels from filenames
def extract_labels_from_filename(filename):
    match = re.search(r'speech_true_(.*)\.wav', filename)
    if match:
        words = match.group(1).split('_')
        return ' '.join(words)
    return ''


# Dataset class
class AudioDataset(Dataset):
    def __init__(self, audio_files, processor):
        self.audio_files = audio_files
        self.processor = processor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path, audio, sr = self.audio_files[idx]
        inputs = self.processor(audio, return_tensors="pt", padding="longest", sampling_rate=sr)
        label = extract_labels_from_filename(os.path.basename(file_path))
        label_ids = self.processor.tokenizer(label, return_tensors="pt").input_ids
        return inputs.input_values.squeeze(), label_ids.squeeze()


# Collate function to handle padding in DataLoader
def collate_fn(batch):
    input_values = [item[0] for item in batch]
    label_ids = [item[1] for item in batch]

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0)
    label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

    return input_values, label_ids


# Load audio files
def load_audio_files(directory):
    audio_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)
                audio_data.append((file_path, y, sr))
    return audio_data


# Training function
def train_model(model, processor, train_loader, num_epochs=3, lr=5e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input_values, label_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(input_values)
            logits = outputs.logits

            # Compute lengths for CTC loss
            input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
            label_lengths = torch.sum(label_ids != -100, dim=1)

            loss = torch.nn.CTCLoss()(logits.transpose(0, 1), label_ids, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

    # Save the model after training
    model.save_pretrained("fine_tuned_wav2vec2")
    processor.save_pretrained("fine_tuned_wav2vec2")


# Inference function with timestamps
def infer_with_timestamps(model, processor, audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    inputs = processor(y, return_tensors="pt", padding="longest", sampling_rate=sr)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Get the frame timestamps
    frame_duration = processor.feature_extractor.chunk_length / sr
    frame_timestamps = [i * frame_duration for i in range(logits.shape[1])]

    # Align the timestamps with the predicted tokens
    word_timestamps = []
    current_word = ""
    current_word_start = None

    for i, token_id in enumerate(predicted_ids[0]):
        token = processor.decode([token_id])
        if token.strip() != "":
            if current_word == "":
                current_word_start = frame_timestamps[i]
            current_word += token
        else:
            if current_word != "":
                word_timestamps.append((current_word, current_word_start, frame_timestamps[i]))
                current_word = ""
                current_word_start = None

    # Handle last word if any
    if current_word != "":
        word_timestamps.append((current_word, current_word_start, frame_timestamps[-1]))

    return transcription, word_timestamps


# Main execution
if __name__ == "__main__":
    data_dir = '../dataset'
    scenes_path = f'{data_dir}/scenes/wav'
    words_path = f'{data_dir}/words'

    scenes_audio = load_audio_files(scenes_path)
    words_audio = load_audio_files(words_path)

    processor, model = load_model_and_tokenizer()

    train_dataset = AudioDataset(scenes_audio, processor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    train_model(model, processor, train_loader, num_epochs=3, lr=5e-5)

    unseen_audio_file = 'path_to_unseen_audio.wav'
    transcription, word_timestamps = infer_with_timestamps(model, processor, unseen_audio_file)

    print("Transcription:", transcription)
    print("Word Timestamps:")
    for word, start, end in word_timestamps:
        print(f"Word: {word}, Start: {start:.2f}s, End: {end:.2f}s")
