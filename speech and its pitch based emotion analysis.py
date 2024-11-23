from flask import Flask, render_template_string
import speech_recognition as sr
import joblib
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import io
import base64

# Load the trained text emotion model
pipe_lr = joblib.load(r'E:\Voice Based Sentiment Analysis\ML_Model\text_emotion.pkl')

# Define emotion dictionary with pitch ranges
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}
pitch_emotion_dict = {
    "anger": (220, 350),
    "disgust": (160, 220),
    "fear": (250, 500),
    "happy": (210, 300),
    "joy": (240, 350),
    "neutral": (85, 180),
    "sad": (75, 120),
    "sadness": (75, 120),
    "shame": (130, 200),
    "surprise": (260, 500)
}

# Initialize Flask
app = Flask(__name__)

# Initialize recognizer
recognizer = sr.Recognizer()

# Function for speech recognition
def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Please Speak Something")

        # Capture audio from the microphone
        audio = recognizer.listen(source)

        try:
            # Recognize speech
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            return None
        except sr.RequestError:
            print("Could not request results; check your internet connection.")
            return None

# Function to extract pitch and predict pitch-based emotion
def analyze_pitch():
    duration = 5  # Record for 5 seconds
    fs = 44100  # Sampling frequency
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    
    # Extract pitch
    audio = audio.flatten()
    pitches, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=fs)
    mean_pitch = np.nanmean(pitches) if len(pitches) > 0 else None

    # Predict emotion based on pitch
    if mean_pitch is not None:
        for emotion, (min_pitch, max_pitch) in pitch_emotion_dict.items():
            if min_pitch <= mean_pitch <= max_pitch:
                return emotion, mean_pitch, audio
    return "unknown", mean_pitch, audio

# Function for emotion prediction based on text
def predict_text_emotion(text):
    result = pipe_lr.predict([text])[0]
    probabilities = pipe_lr.predict_proba([text])[0]
    emoji_icon = emotions_emoji_dict.get(result, "🤔")
    return result, probabilities, emoji_icon

# Generate bar chart for emotion probabilities
def create_bar_chart(probabilities, labels):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, probabilities, color='skyblue')
    plt.title("Emotion Prediction Probabilities")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return image

# Generate wave plot for audio
def create_wave_plot(audio, fs):
    plt.figure(figsize=(10, 3))
    plt.plot(audio, color="blue")
    plt.title("Waveform of Recorded Audio")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return image

# Flask route to display results
@app.route("/")
def home():
    text = speech_to_text()
    pitch_emotion, mean_pitch, audio = analyze_pitch()
    if text and mean_pitch:
        # Predict text-based emotion
        text_emotion, probabilities, text_emoji = predict_text_emotion(text)

        # Compare results
        labels = list(pipe_lr.classes_)
        pitch_emoji = emotions_emoji_dict.get(pitch_emotion, "🤔")
        final_emotion = text_emotion if text_emotion == pitch_emotion else "mean value"

        # Generate plots
        bar_chart = create_bar_chart(probabilities, labels)
        wave_plot = create_wave_plot(audio, 44100)

        # Render HTML
        result_html = f"""
        <html>
            <head>
                <title>Emotion Detector</title>
            </head>
            <body>
                <h1>Emotion Detector</h1>
                <h2>Detected Text Emotion: {text_emotion} {text_emoji}</h2>
                <h2>Detected Pitch Emotion: {pitch_emotion} {pitch_emoji}</h2>
                <h2>Final Emotion: {final_emotion}</h2>
                <p><strong>Mean Pitch:</strong> {mean_pitch:.2f} Hz</p>
                <img src="data:image/png;base64,{wave_plot}" alt="Wave Plot">
                <img src="data:image/png;base64,{bar_chart}" alt="Bar Chart">
            </body>
        </html>
        """
        return result_html
    else:
        return "<p>Could not detect emotion. Please try again.</p>"

if __name__ == "__main__":
    app.run(debug=True)