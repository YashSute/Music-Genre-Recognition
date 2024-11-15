import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt

# Step 1: Data Augmentation
def augment_audio(audio, sr=22050):
    audio_stretch = librosa.effects.time_stretch(audio, rate=0.8)
    audio_shift = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise
    return [audio, audio_stretch, audio_shift, audio_noise]

def load_data_with_augmentation(data_path, genres, sr=22050, duration=30, mono=True):
    X = []
    y = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            try:
                audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=mono)
                augmented_audios = augment_audio(audio, sr)
                for aug_audio in augmented_audios:
                    X.append(aug_audio)
                    y.append(genre)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print("Data loaded with augmentation.")
    return X, y

data_path = 'D:\\Data\\genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']
X, y = load_data_with_augmentation(data_path, genres)
print("Number of samples:", len(X))
print("Number of labels:", len(y))

# Step 2: Feature Extraction
def extract_features(X, sr=22050, n_mfcc=13):
    features = []
    for i, audio in enumerate(X):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        
        combined_features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean])
        features.append(combined_features)
    print("Features extracted.")
    return np.array(features)

X_features = extract_features(X)
print("Shape of features:", X_features.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print("Data split into train and test sets.")

# Step 3: Hyperparameter Tuning with Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units_1', min_value=256, max_value=1024, step=128), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(units=hp.Int('units_2', min_value=256, max_value=1024, step=128), activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(units=hp.Int('units_3', min_value=128, max_value=512, step=64), activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(len(genres), activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=2, directory='my_dir', project_name='music_genre_classification')

tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal number of units in the first layer is {best_hps.get('units_1')},
the second layer is {best_hps.get('units_2')},
the third layer is {best_hps.get('units_3')},
with dropout rates of {best_hps.get('dropout_1')}, {best_hps.get('dropout_2')}, and {best_hps.get('dropout_3')},
and learning rate of {best_hps.get('learning_rate')}.
""")

# Step 4: Model Training with Best Hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2)

y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=genres, yticklabels=genres)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model.save('music_genre_classifier.h5')

# Step 5: Load and Predict
def load_and_predict(model_path, X_test):
    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return y_pred

loaded_model_path = 'improved_music_genre_classifier.h5'
y_pred_loaded = load_and_predict(loaded_model_path, X_test)
loaded_model_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f'Loaded Model Accuracy: {loaded_model_accuracy:.2f}')

