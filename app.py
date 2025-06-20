import streamlit as st
import cv2
import numpy as np
import os
import random
import threading
import time



# Load the face recognizer model and classifier
fishface = cv2.face.FisherFaceRecognizer_create()
fishface.read("model.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotions = ["angry", "happy", "sad", "neutral"]

# Shared thread-safe data
thread_data = {"song_name": "", "song_end_time": 0, "start_requested": False}

# Emotion detection
def detect_emotion(gray, face):
    for (x, y, w, h) in face:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (350, 350))
        pred, conf = fishface.predict(face_img)
        print(f"Detected label: {pred}, Confidence: {conf:.2f}")
        if conf < 3000:
            return emotions[pred]
    return None

# Music playback
def play_song_streamlit(path):
    audio_bytes = open(path, "rb").read()
    st.audio(audio_bytes, format="audio/mp3")
    thread_data["song_name"] = os.path.basename(path)
    thread_data["start_requested"] = False

# UI Setup
st.set_page_config(page_title="Emotion Music Player", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1487215078519-e21cc028cb29?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    .main {
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emotion-box {
        background-color: #ffffffdd;
        border: 2px solid #ccc;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color:#4CAF50;'>🎧 Emotion-Based Music Player</h1>
""", unsafe_allow_html=True)
FRAME_WINDOW = st.image([], use_container_width=True)
detected_emotion = st.empty()
now_playing = st.empty()
st.markdown("---")

# Load music library
music_library = {emotion: os.listdir(f"songs/{emotion}") if os.path.exists(f"songs/{emotion}") else [] for emotion in emotions}

st.subheader("🎶 Available Songs")
cols = st.columns(len(emotions))
for i, emotion in enumerate(emotions):
    with cols[i]:
        st.markdown(f"<div class='emotion-box'><strong>{emotion.capitalize()}</strong>", unsafe_allow_html=True)
        for song in music_library[emotion]:
            st.markdown(f"<li>{song}</li>", unsafe_allow_html=True)
        if not music_library[emotion]:
            st.markdown("_No songs available_")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Control Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🛑 Stop Music"):
        pygame.mixer.music.stop()
        thread_data["song_name"] = ""
        thread_data["song_end_time"] = 0
with col2:
    if st.button("🎬 Start Detection"):
        thread_data["start_requested"] = True

loading_spinner = st.empty()

# Detection function
cap = None
last_emotion = None

def run_detection():
    with loading_spinner.container():
        st.markdown("""
        <div style='position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(255,255,255,0.85); z-index: 1000; display: flex; align-items: center; justify-content: center;'>
            <div style='text-align: center;'>
                                <h3 style='color: #4CAF50;'>🔍 Detecting Emotion...</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    global cap, last_emotion
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    emotion = detect_emotion(gray, face)
    if emotion and emotion != last_emotion:
        last_emotion = emotion
        detected_emotion.markdown(f"### 😃 Detected Emotion: **{emotion.upper()}**")
        songs = music_library.get(emotion, [])
        if songs:
            selected_song = random.choice(songs)
            song_path = os.path.join("songs", emotion, selected_song)
            cap.release()
            cap = None
            FRAME_WINDOW.empty()
            threading.Thread(target=play_song, args=(song_path,), daemon=True).start()
            return
        else:
            now_playing.markdown("No song found for detected emotion.")

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    loading_spinner.empty()
    if cap:
        cap.release()

# Monitor loop
def monitor():
    while True:
        if pygame.mixer.music.get_busy():
            remaining = int(thread_data["song_end_time"] - time.time())
            now_playing.markdown(f"**🎵 Now Playing:** {thread_data['song_name']} ({remaining} sec remaining)" if remaining > 0 else "Ready for next emotion...")
            time.sleep(0.3)
        elif thread_data["start_requested"]:
            run_detection()
        else:
            time.sleep(0.05)  # Faster refresh

if st.button("🎬 Start Detection"):
    run_detection()


