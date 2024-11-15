import os
import numpy as np
import librosa
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox

# Define the path to the saved model
model_path = r'D:\Python\Its_new\Its_new\music_genre_classifier.h5'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Compile the model with a placeholder loss and metric to suppress the warning
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the genres 
genres = ['blues', 'classical', 'country', 'disco', 'hiphop','metal', 'pop', 'reggae', 'rock']

# Function to extract features from an audio file
def extract_features_from_audio(file_path, sr=22050, n_mfcc=13, duration=30, mono=True):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=mono)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Function to browse for an audio file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        audio_file_path.set(file_path)

# Function to classify the audio file
def classify_audio():
    file_path = audio_file_path.get()
    if not file_path:
        messagebox.showerror("Error", "Please select an audio file first")
        return
    
    try:
        features = extract_features_from_audio(file_path)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        predicted_genre_index = np.argmax(model.predict(features), axis=1)[0]
        predicted_genre = genres[predicted_genre_index]
        result.set(f'The predicted genre of the audio file is: {predicted_genre}')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the file: {e}")

# Initialize the main window
root = tk.Tk()
root.title("Music Genre Classifier")

# StringVar to hold the path of the audio file and the result
audio_file_path = tk.StringVar()
result = tk.StringVar()

# Create and place the widgets in the window
tk.Label(root, text="Select an audio file:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=audio_file_path, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=10)

tk.Button(root, text="Classify", command=classify_audio).grid(row=1, column=1, pady=20)
tk.Label(root, textvariable=result).grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Run the application
root.mainloop()
