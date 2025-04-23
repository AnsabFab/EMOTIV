import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
from deepface import DeepFace
import threading

# Set page config
st.set_page_config(
    page_title="Emotion Analysis - Debug Version",
    page_icon="😀",
    layout="wide"
)

# Initialize session state variables
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = pd.DataFrame(columns=['timestamp', 'emotion', 'valence', 'arousal'])
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# Title and description
st.title("Emotion Analysis - Debug Version")
st.markdown("""
This is a simplified version of the emotion analysis app for debugging purposes.
""")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    # Camera feed
    st.subheader("Camera Feed")
    camera_placeholder = st.empty()
    
    # Debug info
    st.subheader("Debug Info")
    debug_placeholder = st.empty()
    
    # Camera device selection
    camera_devices = [0, 1, 2]  # Most systems will have camera 0
    selected_camera = st.selectbox("Select Camera", options=camera_devices, index=0)
    
    # Start/Stop button
    start_button = st.button("Start/Stop Analysis")

with col2:
    # Metrics display
    st.subheader("Detected Emotion")
    emotion_placeholder = st.empty()
    
    # Data table
    st.subheader("Data Log")
    data_placeholder = st.empty()

# Function to analyze facial features and emotions
def analyze_face(frame):
    try:
        debug_placeholder.text("Analyzing face...")
        # Use DeepFace for emotion analysis
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract emotion data
        emotions = analysis['emotion']
        dominant_emotion = analysis['dominant_emotion']
        
        # Map emotions to valence and arousal based on psychological models
        emotion_mapping = {
            'happy': {'valence': 0.9, 'arousal': 0.7},
            'sad': {'valence': 0.2, 'arousal': 0.3},
            'angry': {'valence': 0.2, 'arousal': 0.9},
            'fear': {'valence': 0.3, 'arousal': 0.8},
            'disgust': {'valence': 0.2, 'arousal': 0.6},
            'surprise': {'valence': 0.7, 'arousal': 0.8},
            'neutral': {'valence': 0.5, 'arousal': 0.4}
        }
        
        # Get valence and arousal for dominant emotion
        valence = emotion_mapping.get(dominant_emotion, {'valence': 0.5})['valence']
        arousal = emotion_mapping.get(dominant_emotion, {'arousal': 0.5})['arousal']
        
        debug_placeholder.text(f"Analysis complete. Dominant emotion: {dominant_emotion}")
        
        return {
            'emotion': dominant_emotion,
            'emotions': emotions,
            'valence': valence,
            'arousal': arousal
        }
    except Exception as e:
        debug_placeholder.text(f"Error in face analysis: {str(e)}")
        return None

# Handle button clicks
if start_button:
    st.session_state.is_analyzing = not st.session_state.is_analyzing
    if st.session_state.is_analyzing:
        debug_placeholder.text("Starting analysis...")
    else:
        debug_placeholder.text("Analysis stopped")

# Main loop
try:
    # Initialize camera
    debug_placeholder.text(f"Initializing camera {selected_camera}...")
    cap = cv2.VideoCapture(selected_camera)
    
    if not cap.isOpened():
        st.error(f"Error: Cannot open camera {selected_camera}")
        debug_placeholder.text(f"Failed to open camera {selected_camera}")
    else:
        debug_placeholder.text(f"Camera {selected_camera} initialized successfully")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            debug_placeholder.text(f"Failed to capture video from camera {selected_camera}")
            st.error(f"Failed to capture video. Please check camera settings.")
            time.sleep(1)
            continue
        
        # Display camera feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Perform analysis if enabled
        if st.session_state.is_analyzing:
            # Analyze face
            face_data = analyze_face(frame)
            
            if face_data:
                # Update metrics display
                emotion_placeholder.markdown(f"""
                ### Detected Emotion: **{face_data['emotion']}**
                - Valence: {face_data['valence']:.2f}
                - Arousal: {face_data['arousal']:.2f}
                
                **All Emotions:**
                {', '.join([f"{emotion}: {score:.1f}%" for emotion, score in face_data['emotions'].items()])}
                """)
                
                # Add to dataframe
                current_time = time.time()
                elapsed_time = current_time - st.session_state.start_time
                
                new_row = pd.DataFrame([{
                    'timestamp': elapsed_time,
                    'emotion': face_data['emotion'],
                    'valence': face_data['valence'],
                    'arousal': face_data['arousal']
                }])
                
                st.session_state.metrics_data = pd.concat([st.session_state.metrics_data, new_row], ignore_index=True)
                
                # Update data table
                data_placeholder.dataframe(st.session_state.metrics_data.round(2), use_container_width=True)
            
            # Sleep to reduce analysis frequency
            time.sleep(1)
        else:
            # Sleep to control refresh rate
            time.sleep(0.05)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    debug_placeholder.text(f"Critical error: {str(e)}")
finally:
    # Clean up resources
    if 'cap' in locals() and cap is not None:
        cap.release()
        debug_placeholder.text("Camera released")
