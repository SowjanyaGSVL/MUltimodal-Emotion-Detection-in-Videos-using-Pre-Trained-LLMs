import streamlit as st

def display_introduction():
    st.title("Multimodal Emotion Detection in Videos using Pre-trained LLM")

    st.markdown("""
        This application is designed to predict emotions from various types of inputs:
        - **Text**: Enter a piece of text to analyze its emotion.
        - **Audio**: Upload an audio file to detect emotions based on the audio content.
        - **Video**: Upload a video file to extract audio and video features for emotion detection.

        The system uses a combination of deep learning models and feature extraction techniques to provide accurate emotion predictions.

        **How It Works:**
        - For text, an LSTM model predicts the emotion based on the provided text.
        - For audio, a CNN model analyzes audio features to determine the emotion.
        - For video, the application extracts text, audio and video features, then uses a language model to predict the most dominant emotion.

        **Features:**
        - Text analysis with LSTM
        - Audio feature extraction using Librosa
        - Video feature extraction using ResNet
        - Integration of multiple data sources for comprehensive emotion prediction

        Feel free to navigate through the options in the sidebar to test the system with your own data.
    """)