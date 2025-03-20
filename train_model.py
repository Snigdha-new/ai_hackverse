import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
print("Hello! The training has started.")


# Load the extracted features
df = pd.read_pickle('audio_features.pkl')

# Convert to NumPy arrays
X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'].tolist())

# Encode emotion labels (convert to numbers)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(40,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('emotion_model.h5')
print("✅ Model trained and saved as 'emotion_model.h5'")
