import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Emotion AI - Speech", page_icon="🎙️")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.write("**Algorithm:** CNN (Conv1D)")
    st.write("**Input:** MFCC features (40)")
    st.write("**Classes:** angry, calm, disgust, fearful, happy, neutral, sad, surprised")
    st.write("**Test Accuracy:** 88.19%")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write("This classifier predicts emotion from speech audio using MFCC features and a CNN trained on the RAVDESS dataset. Built for CodeAlpha ML Internship by Jayesh Kadam.")

st.markdown("## 🎭 Emotion AI: Speech Emotion Classifier")
st.write("Upload a speech audio clip, and let the model read the emotion.")

emoji_map = {
    'angry': '😠', 'calm': '😌', 'disgust': '🤢', 'fearful': '😨',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprised': '😲'
}

@st.cache_resource
def get_model():
    return load_model("emotion_recognition_model.h5")

model = get_model()
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

uploaded_file = st.file_uploader("Upload audio — WAV, MP3", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button("🔮 Predict Emotion"):
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        y, sr = librosa.load("temp_audio.wav", duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0).reshape(1, 40, 1)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        emotion = emotions[predicted_class]

        st.markdown(f"""
        <div style='text-align:center; padding:20px; background:#f0f2f6; border-radius:10px;'>
            <div style='font-size:60px;'>{emoji_map[emotion]}</div>
            <div style='font-size:24px; font-weight:bold;'>{emotion.capitalize()}</div>
            <div style='color:gray;'>{confidence:.2f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Upload an audio file to get started.")
