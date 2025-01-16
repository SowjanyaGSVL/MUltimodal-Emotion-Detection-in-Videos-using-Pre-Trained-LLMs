import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import joblib
import torchvision.models as models
import speech_recognition as sr
from pydub import AudioSegment
import os
import librosa
import moviepy.editor as mp
import re
import ast

# Load the LabelEncoder
label_encoder = joblib.load(r'C:\Users\datapro\Notebooks\LLM\label_encoder.pkl')
audio_output_path = r'C:\Users\datapro\Notebooks\temp_audio.wav'
video_path = "temp_video.mp4"

# Define the MultimodalModel class
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.audio_fc = nn.Linear(13, 64)
        self.video_fc = nn.Linear(1*6*6, 64)  # Adjust input size based on video feature dimensions
        self.embedding_fc = nn.Linear(4096, 128)
        self.combined_fc = nn.Linear(64 + 64 + 128, 128)
        self.output = nn.Linear(128, 7)  # Assuming 7 emotion classes

    def forward(self, audio_features, video_features, embeddings):
        print("Audio features shape:", audio_features.shape)
        print("Video features shape before flattening:", video_features.shape)
        
        audio_out = F.relu(self.audio_fc(audio_features))
        video_out = video_features.view(video_features.size(0), -1)  # Flatten video features
        print("Video features shape after flattening:", video_out.shape)
        
        video_out = F.relu(self.video_fc(video_out))
        embedding_out = F.relu(self.embedding_fc(embeddings))
        combined_out = torch.cat((audio_out, video_out, embedding_out), dim=1)
        print("Combined features shape:", combined_out.shape)
        
        combined_out = F.relu(self.combined_fc(combined_out))
        output = self.output(combined_out)
        return output

# Load the saved model
model_path = r'C:\Users\datapro\Notebooks\LLM\multimodal_model.pth'
model = MultimodalModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load the ResNet18 model for video feature extraction
resnet18_model = models.resnet18(weights='DEFAULT')
resnet18_model.eval()

# Define a transform to resize and normalize video frames
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize session state
if 'video_features' not in st.session_state:
    st.session_state.video_features = None
if 'audio_features' not in st.session_state:
    st.session_state.audio_features = None
if 'text_embedding' not in st.session_state:
    st.session_state.text_embedding = None
if 'text' not in st.session_state:
    st.session_state.text = None

def get_embedding(utterance):
    try:
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'llama3', 'prompt': utterance}
        )
        response.raise_for_status()
        result = response.json()
        if 'embedding' in result:
            return result['embedding']
        else:
            st.error("Embedding not found in response.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed for utterance '{utterance}': {e}")
        return None

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load ResNet model
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(frame)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            feature = resnet18(input_tensor)
        
        features.append(feature.squeeze(0))  # Remove batch dimension

    cap.release()
    
    if not features:
        features = [torch.zeros(1000)]  # Handle empty case

    features = torch.stack(features)  # Shape: [num_frames, 1000]
    num_frames = features.size(0)
    feature_size = features.size(1)
    
    print("Features shape after stacking:", features.shape)
    print("Number of frames:", num_frames)
    print("Feature size:", feature_size)
    
    # Extract first 3 and last 3 tensors
    if len(features) < 6:
        raise ValueError("Not enough frames to extract the first 3 and last 3 tensors")
    
    first_3_tensors = features[:3]
    last_3_tensors = features[-3:]
    
    # Combine first 3 and last 3 tensors
    combined_features = torch.cat((first_3_tensors, last_3_tensors), dim=0)
    
    # Flatten combined features
    combined_features = combined_features.view(-1)  # Flatten to 1D tensor
    
    # Ensure the combined features tensor has exactly 36 elements
    target_size = 36
    if combined_features.size(0) > target_size:
        combined_features = combined_features[:target_size]  # Truncate
    elif combined_features.size(0) < target_size:
        combined_features = torch.cat([combined_features, torch.zeros(target_size - combined_features.size(0))])  # Pad
    
    # Reshape to [1, 1, 6, 6]
    combined_features = combined_features.view(1, 1, 6, 6)
    
    return combined_features

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        st.error(f"Error extracting features from {audio_path}: {str(e)}")
        return None

def extract_audio_from_video(video_path, audio_output_path):
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path)
        
        # Extract audio features
        audio_features = extract_audio_features(audio_output_path)
        return audio_features
    except Exception as e:
        st.error(f"Error processing video {video_path}: {str(e)}")
        return None

def extract_text_from_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_file(audio_path)
        wav_path = f"{os.path.splitext(audio_path)[0]}.wav"
        audio.export(wav_path, format="wav")
        
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        #st.error(f"Error extracting text from audio {audio_path}: {str(e)}")
        return "Error extracting text from audio"

def pad_frames(frames, max_length, padding_value=0):
    if len(frames) < max_length:
        padding_frames = [torch.full_like(frames[0], padding_value) for _ in range(max_length - len(frames))]
        frames.extend(padding_frames)
    return frames

def string_to_tensor(tensor_string):
    tensor_string = tensor_string.strip('tensor()')
    tensor_string = tensor_string.replace('\n', '').replace(' ', '')
    tensor_string = tensor_string.replace('...', '')
    tensor_string = re.sub(r',+', ',', tensor_string)
    tensor_string = re.sub(r',(\s*\])', r'\1', tensor_string)
    tensor_string = re.sub(r'\[\s*,', '[', tensor_string)
    tensor_string = re.sub(r',\s*\]', ']', tensor_string)
    tensor_string = f'[{tensor_string}]'
    
    try:
        tensor_list = eval(tensor_string)
    except (SyntaxError, ValueError) as e:
        st.error(f"Error in parsing tensor string: {e}")
        return torch.empty((0, 0))
    
    tensor = torch.tensor(tensor_list, dtype=torch.float32)
    return tensor

def convert_features_to_tensors(df):
    def safe_literal_eval(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []

    audio_features_list = [safe_literal_eval(f) for f in df['Audio Features'].values]
    embeddings_list = [safe_literal_eval(f) for f in df['Embedding'].values]
    video_features_list = [safe_literal_eval(f) for f in df['Video_Features'].values]

    audio_features_tensor = torch.stack([string_to_tensor(f) for f in audio_features_list])
    embeddings_tensor = torch.stack([string_to_tensor(f) for f in embeddings_list])
    video_features_tensor = torch.stack([string_to_tensor(f) for f in video_features_list])

    return audio_features_tensor, embeddings_tensor, video_features_tensor

def predict_emotion_audio_video_text(audio_features, video_features, text_embedding):
    # Ensure the input tensors are of the expected shape
    audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    video_features = torch.tensor(video_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(audio_features, video_features, text_embedding)
        _, predicted_class = torch.max(output, 1)
    
    # Decode the predicted class index to emotion
    emotion = label_encoder.inverse_transform([predicted_class.item()])[0]
    return emotion

def predict_emotion_text(text):
    # Load the tokenizer
    tokenizer = joblib.load(r'C:\Users\datapro\Notebooks\Text\tokenizer.pkl')
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Load the LSTM model
    lstm_model = joblib.load(r'C:\Users\datapro\Notebooks\Text\text_lstm_emotion_model.pth')
    
    with torch.no_grad():
        output = lstm_model(input_ids, attention_mask)
        _, predicted_class = torch.max(output, 1)
    
    # Decode the predicted class index to emotion
    emotion = label_encoder.inverse_transform([predicted_class.item()])[0]
    return emotion

st.title("Emotion Detection")

# Select model to use
model_option = st.sidebar.selectbox("Choose model", ["Multimodal Emotion Prediction", "Text Emotion Detection"])

if model_option == "Multimodal Emotion Prediction":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if video_file:
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        st.session_state.video_features = extract_video_features(video_path)
        st.session_state.audio_features = extract_audio_from_video(video_path, audio_output_path)
        st.session_state.text = extract_text_from_audio(audio_output_path)
        st.session_state.text_embedding = get_embedding(st.session_state.text)
        
        if st.session_state.video_features is not None and st.session_state.audio_features is not None and st.session_state.text_embedding is not None:
            emotion = predict_emotion_audio_video_text(st.session_state.audio_features, st.session_state.video_features, st.session_state.text_embedding)
            st.write(f"Predicted Emotion: {emotion}")

elif model_option == "Text Emotion Detection":
    text_input = st.text_area("Enter text")
    
    if text_input:
        st.session_state.text = text_input
        st.session_state.text_embedding = get_embedding(text_input)
        
        if st.session_state.text_embedding is not None:
            emotion = predict_emotion_text(text_input)
            st.write(f"Predicted Emotion: {emotion}")
