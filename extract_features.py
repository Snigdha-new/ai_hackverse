import librosa
import numpy as np
import os
import pandas as pd

# Extract audio features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Path to dataset
dataset_path = 'Audio_Speech_Actors_01-24'
data = []

# Emotion labels from RAVDESS filenames
emotion_labels = {
    '01': 'Neutral',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fearful'
}

# Extract features from all audio files
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            emotion_code = file.split('-')[2]
            if emotion_code in emotion_labels:
                emotion = emotion_labels[emotion_code]
                features = extract_features(os.path.join(root, file))
                data.append([features, emotion])

# Save extracted features
df = pd.DataFrame(data, columns=['Features', 'Emotion'])
df.to_pickle('audio_features.pkl')
print("✅ Features extracted and saved as 'audio_features.pkl'")
