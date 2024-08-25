from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import librosa
import joblib
import secrets
from tensorflow.keras.models import load_model

# Define the Flask app with specified template and static folders
app = Flask(
    __name__,
    template_folder="./templates",
    static_folder="./static",
)

# Set a secret key for securely signing the session data
app.secret_key = secrets.token_hex(16)

# Load the pre-trained LSTM model and scaler
lstm_model_path = os.path.join(app.root_path, 'lstm_model.h5')
scaler_path = os.path.join(app.root_path, 'scaler.pkl')

lstm_model = load_model(lstm_model_path)
scaler = joblib.load(scaler_path)

# Constants for feature extraction
n_mfcc = 13
max_timesteps = 130  # Adjust this based on your model's training data

# Helper function to extract features from audio files
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)  # Load audio with a fixed duration
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose to have shape (timesteps, features)

# Helper function to pad features
def pad_features(features, max_timesteps, n_mfcc):
    num_timesteps = features.shape[0]
    if num_timesteps < max_timesteps:
        padded_features = np.zeros((max_timesteps, n_mfcc))
        padded_features[:num_timesteps, :] = features
    else:
        padded_features = features[:max_timesteps, :]
    return padded_features

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Login page route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simulate successful login (add authentication logic here if needed)
        return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            # Save the file temporarily in the 'uploads' directory
            uploads_dir = os.path.join(app.root_path, 'uploads')
            filepath = os.path.join(uploads_dir, file.filename)
            file.save(filepath)

            try:
                # Extract features and classify the genre
                features = extract_features(filepath)
                
                # Pad features to match the input shape required by the LSTM model
                feature_vector_padded = pad_features(features, max_timesteps, n_mfcc)
                
                # Reshape and scale features
                feature_vector_reshaped = feature_vector_padded.reshape(-1, n_mfcc)
                feature_vector_scaled = scaler.transform(feature_vector_reshaped)
                feature_vector_scaled = feature_vector_scaled.reshape((1, max_timesteps, n_mfcc))
                
                # Predict genre using the LSTM model
                prediction = lstm_model.predict(feature_vector_scaled)
                predicted_class = np.argmax(prediction)
                
                # Map the index to the actual genre label (modify the labels according to your dataset)
                genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                genre = genre_labels[predicted_class]

                # Return the predicted genre as a JSON response
                return jsonify({"predicted_genre": genre}), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

            finally:
                # Optionally, delete the file after processing
                os.remove(filepath)

    return render_template('upload.html')

# Route for redirecting to the home page
@app.route('/home')
def home():
    return redirect(url_for('index'))

# Route for handling a successful logout (optional)
@app.route('/logout')
def logout():
    # Add logout logic here if needed
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    uploads_dir = os.path.join(app.root_path, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    app.run(debug=True)
