# Voice-Based-Sentiment-Analysis

## Overview
This project is a voice-based sentiment analysis application built using Flask and speech recognition, with a pre-trained model to detect emotions from spoken words. The application captures audio input, converts it to text, predicts the emotion, and provides a confidence score with emoji representation. It’s an interactive tool that can serve in applications like customer service analysis, mood detection, and more.

## Features
Speech-to-Text Conversion: Uses Google’s speech recognition to capture and transcribe audio.
Emotion Prediction: Utilizes a trained model (text_emotion.pkl) to classify emotions from text.
Emoji Representation: Displays emotions with emojis for enhanced user experience.
Confidence Score: Shows the confidence level of the prediction.

## Demo
After starting the server, the application prompts the user to speak. It then displays the detected text, predicted emotion, corresponding emoji, and confidence percentage.
Here’s a quick demo of the Voice-Based Sentiment Analysis application in action:

![Demo Video]("Media\sentiment analysis.mp4")

## Installation

### Clone the repository:
   git clone
### Navigate to the project directory:
   cd voice-emotion-detector
### Install dependencies:
   pip install -r requirements.txt
   Ensure text_emotion.pkl is in the project directory.

## Usage

### Run the Flask application:
python app.py
### Use the Application:
1) Access the application at localhost in a web browser.
2) Speak into your microphone when prompted.
3) View the recognized text, predicted emotion, emoji, and confidence.
   
## Code Explanation
1) Speech_to_text: Converts spoken input into text using Google’s speech recognition.
2) predict_emotion: Uses a pre-trained model to predict the emotion of the transcribed text.
3) Flask Route:
  Home Route (/): Displays detected text, emotion, emoji, and confidence percentage.

## Model Information
The model text_emotion.pkl was trained on text data to classify emotions such as anger, joy, sadness, and surprise. It’s integrated with a Flask app to provide real-time feedback from voice inputs.

## Example Results

Text Input	Predicted Emotion	Emoji	Confidence
"I'm so happy!"	Joy	🤗	95%
"I am worried"	Fear	😨	89%


## Requirements
1) Flask
2) SpeechRecognition
3) Joblib
4) Numpy
5) Pandas

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to all contributors and the open-source community!
