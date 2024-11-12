from flask import Flask, render_template_string
import speech_recognition as sr
import joblib
import numpy as np

# Load the trained model
pipe_lr = joblib.load("text_emotion.pkl")

# Define emotion dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}


recognizer = sr.Recognizer()

# Initialize Flask
app = Flask(__name__)

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

# Function for emotion prediction
def predict_emotion(text):
    result = pipe_lr.predict([text])[0]
    probability = pipe_lr.predict_proba([text])
    emoji_icon = emotions_emoji_dict.get(result, "🤔")  # default icon if not in dictionary
    confidence = np.max(probability) * 100  # Convert to percentage
    print(f"Emotion: {result} {emoji_icon} with {confidence:.2f}% confidence.")
    return result, confidence, emoji_icon

# Flask route to display results
@app.route("/")
def home():
    text = speech_to_text()
    if text:
        emotion, confidence, emoji = predict_emotion(text)
        result_html = f"""
        <html>
            <head>
                <title>Emotion Detector</title>
            </head>
            <body>
                <h1>Voice Emotion Detector</h1>
                <p><strong>Detected Text:</strong> {text}</p>
                <p><strong>Detected Emotion:</strong> {emotion} {emoji}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </body>
        </html>
        """
        return result_html
    else:
        return "<p>Could not recognize speech. Please try again.</p>"

if __name__ == "__main__":
    app.run(debug=True)
