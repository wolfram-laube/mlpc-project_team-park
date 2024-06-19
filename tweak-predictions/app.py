from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/wav_files'  # Directory for WAV files

target_commands = [
    "ofen an", "ofen aus", "alarm an", "alarm aus", "lüftung an", "lüftung aus",
    "heizung an", "heizung aus", "licht an", "licht aus", "fernseher an",
    "fernseher aus", "staubsauger an", "staubsauger aus", "radio an", "radio aus",
    "(none)"
]

def load_predictions():
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.csv')
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['filename', 'command', 'timestamp'])
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df

def save_predictions(df):
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.csv')
    df.to_csv(csv_path, index=False)

@app.route('/')
def index():
    df = load_predictions()
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.wav')]
    new_entries = []
    for file in files:
        if file.split('.')[0] not in df['filename'].values:
            new_entries.append({'filename': file.split('.')[0], 'command': '', 'timestamp': ''})
    if new_entries:
        df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
        save_predictions(df)
    return render_template('index.html', predictions=df, commands=target_commands)

@app.route('/update', methods=['POST'])
def update():
    filename = request.form['filename']
    command = request.form['command']
    timestamp = request.form['timestamp']
    df = load_predictions()
    if command == "(none)":
        df = df[df['filename'] != filename]
    else:
        df.loc[df['filename'] == filename, ['command', 'timestamp']] = [command, timestamp]
    save_predictions(df)
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
