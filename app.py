import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Define emotions (adjust if needed)
emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful']

# Function to extract features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Predict emotion
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return emotions[np.argmax(prediction)]

# Streamlit UI
st.title("🎵 Emotion Detection from Voice")
st.write("Upload an audio file to detect the emotion.")

audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    predicted_emotion = predict_emotion("temp_audio.wav")
    st.success(f"Detected Emotion: {predicted_emotion}")
