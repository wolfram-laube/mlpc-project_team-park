import os
from gtts import gTTS
from tqdm import tqdm
import random
import librosa
import numpy as np
from pydub import AudioSegment

# List of random short phrases or sentences that do not correspond to any command
random_phrases = [
    "Guten Tag", "Wie geht es Ihnen", "Was machen Sie", "Schönes Wetter heute", "Ich habe Hunger",
    "Ich bin müde", "Lass uns gehen", "Was ist das", "Wie spät ist es", "Ich verstehe nicht",
    "Wo ist der Bahnhof", "Haben Sie die Uhrzeit", "Können Sie mir helfen", "Das ist interessant",
    "Ich bin verloren", "Was kostet das", "Wie viel Uhr ist es", "Wo ist das nächste Geschäft",
    "Ich brauche Hilfe", "Das ist eine gute Idee", "Können Sie das wiederholen", "Ich habe keine Ahnung",
    "Wohin gehen Sie", "Was wollen Sie trinken", "Ich muss los", "Wo wohnen Sie", "Haben Sie Geschwister",
    "Was ist Ihr Beruf", "Wie alt sind Sie", "Was haben Sie gesagt"
]

# List of German speakers (GTTS allows different accents and speeds)
speakers = ["de", "de-DE", "de-AT", "de-CH"]

# Directory to save the generated audio files
output_dir = './unknown_commands'
os.makedirs(output_dir, exist_ok=True)

# Generate audio files for each random phrase
num_commands = 500
print("Generating unknown command audio samples...")
for i in tqdm(range(num_commands)):
    phrase = random.choice(random_phrases)
    speaker = random.choice(speakers)
    tts = gTTS(text=phrase, lang=speaker)
    mp3_path = os.path.join(output_dir, f'unknown_command_{i}.mp3')
    wav_path = os.path.join(output_dir, f'unknown_command_{i}.wav')
    tts.save(mp3_path)

    # Convert mp3 to wav
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    os.remove(mp3_path)  # Remove the mp3 file after conversion

print(f"{num_commands} unknown command audio samples generated.")


# Function to extract noise clips from a given file
def extract_noise_clips(file_path, output_dir, clip_duration, max_clips):
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    clip_samples = int(clip_duration * sr)

    for i in range(max_clips):
        start = np.random.randint(0, len(y) - clip_samples)
        end = start + clip_samples
        clip = y[start:end]
        output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(file_path))[0]}_clip_{i}.wav')
        librosa.output.write_wav(output_path, clip, sr)


# Directory containing ESC-50 dataset
esc50_dir = './noises'
noise_files = [f for f in os.listdir(esc50_dir) if f.endswith('.wav')]
desired_clips = 500
clip_duration = 1.0  # Duration of each clip in seconds

print("Generating noise clips...")
clips_extracted = 0
for file in tqdm(noise_files):
    if clips_extracted >= desired_clips:
        break
    extract_noise_clips(os.path.join(esc50_dir, file), output_dir, clip_duration,
                        max_clips=(desired_clips - clips_extracted))
    clips_extracted += desired_clips - clips_extracted

print(f"{clips_extracted} noise clips extracted.")
