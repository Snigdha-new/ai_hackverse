import numpy as np
import librosa
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = keras.models.load_model('emotion_model.h5')

# Emotion labels
encoder = LabelEncoder()
encoder.classes_ = np.array(['Angry', 'Fearful', 'Happy', 'Neutral', 'Sad'])

# Function to extract features from a new audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Predict emotion from audio
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)
    emotion = encoder.inverse_transform([np.argmax(prediction)])
    print(f"🎵 Predicted Emotion: {emotion[0]}")

# Test with your own audio file (rename your file to 'test_audio2.wav')
predict_emotion('test_audio2.wav')
