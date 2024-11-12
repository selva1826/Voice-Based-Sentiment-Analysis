import speech_recognition as sr
import joblib
import numpy as np
import pandas as pd


pipe_lr = joblib.load("text_emotion.pkl")


emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}


recognizer = sr.Recognizer()


def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Please Speak Something")

        
        audio = recognizer.listen(source)

        try:
            
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            return None
        except sr.RequestError:
            print("Could not request results; check your internet connection.")
            return None

def predict_emotion(text):
    result = pipe_lr.predict([text])[0]
    probability = pipe_lr.predict_proba([text])
    emoji_icon = emotions_emoji_dict.get(result, "🤔")  
    confidence = np.max(probability) * 100
    print(f"Emotion: {result} {emoji_icon} with {confidence:.2f}% confidence.")
    return result, confidence, emoji_icon


def main():
    text = speech_to_text()
    if text:
        emotion, confidence, emoji = predict_emotion(text)
        print(f"Detected Emotion: {emotion} {emoji}, Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
