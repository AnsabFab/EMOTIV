import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import mediapipe as mp
from deepface import DeepFace
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Emotion & Personality Analysis",
    page_icon="😀",
    layout="wide"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define the metrics we'll track
metrics = [
    "valence", "arousal", "dominance", "cognitive_load", 
    "emotional_stability", "openness", "agreeableness", 
    "neuroticism", "conscientiousness", "extraversion", 
    "stress_index", "engagement_level"
]

# Initialize session state variables
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = pd.DataFrame(columns=['timestamp'] + metrics)
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = np.array([])
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'face_data' not in st.session_state:
    st.session_state.face_data = None

# Title and description
st.title("Real-time Emotion & Personality Analysis")
st.markdown("""
This application analyzes facial expressions and voice patterns to estimate psychological metrics in real-time:
- Emotional dimensions (valence, arousal, dominance)
- Personality traits (openness, agreeableness, neuroticism, conscientiousness, extraversion)
- Cognitive measures (cognitive load, emotional stability, stress index, engagement level)
""")

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    # Camera feed and controls
    st.subheader("Camera Feed")
    camera_placeholder = st.empty()
    
    # Audio recording controls
    st.subheader("Audio Controls")
    audio_col1, audio_col2 = st.columns(2)
    
    with audio_col1:
        start_button = st.button("Start Recording")
    
    with audio_col2:
        stop_button = st.button("Stop Recording")
    
    # Camera device selection
    st.subheader("Settings")
    camera_devices = [0, 1, 2, 3]  # Most systems will have camera 0 at minimum
    selected_camera = st.selectbox("Select Camera", options=camera_devices, index=0)
    
    # Sample rate for audio
    sample_rate = st.slider("Audio Sample Rate", min_value=8000, max_value=48000, value=16000, step=8000)
    
    # Analysis frequency
    analysis_freq = st.slider("Analysis Frequency (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

with col2:
    # Metrics display
    st.subheader("Real-time Metrics")
    metrics_placeholder = st.empty()
    
    # Graphs
    st.subheader("Metrics History")
    history_placeholder = st.empty()
    
    # Data table
    st.subheader("Data Log")
    data_placeholder = st.empty()


# Audio callback function
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    st.session_state.audio_queue.put(indata.copy())


# Function to start audio recording
def start_audio_recording():
    st.session_state.audio_data = np.array([])
    st.session_state.is_recording = True
    
    def record_audio():
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
            while st.session_state.is_recording:
                if not st.session_state.audio_queue.empty():
                    audio_chunk = st.session_state.audio_queue.get()
                    st.session_state.audio_data = np.append(st.session_state.audio_data, audio_chunk)
                time.sleep(0.1)
    
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()


# Function to stop audio recording
def stop_audio_recording():
    st.session_state.is_recording = False
    if len(st.session_state.audio_data) > 0:
        write("recorded_audio.wav", sample_rate, st.session_state.audio_data)


# Function to extract facial landmarks
def extract_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


# Function to calculate eye aspect ratio (EAR)
def calculate_ear(landmarks, face_landmarks):
    # MediaPipe face mesh indices for the eyes
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    if not landmarks:
        return 0.0
    
    # Get left eye landmarks
    left_eye_points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                               for point in LEFT_EYE])
    
    # Get right eye landmarks
    right_eye_points = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                                for point in RIGHT_EYE])
    
    # Calculate the EAR for both eyes
    def eye_aspect_ratio(eye_points):
        # Compute the euclidean distances between the vertical eye landmarks
        vert1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vert2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        horiz = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (vert1 + vert2) / (2.0 * horiz)
        return ear
    
    # Use approximations for EAR calculation with MediaPipe points
    left_ear = eye_aspect_ratio(left_eye_points[[0, 4, 8, 12, 14, 10]])
    right_ear = eye_aspect_ratio(right_eye_points[[0, 4, 8, 12, 14, 10]])
    
    # Average the EAR for both eyes
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear


# Extract audio features
def extract_audio_features(audio_data, sr):
    if len(audio_data) < sr:
        # Not enough data, return default values
        return {
            "pitch_mean": 0,
            "pitch_std": 0,
            "energy": 0,
            "tempo": 0,
            "spectral_centroid": 0,
            "spectral_bandwidth": 0,
            "spectral_rolloff": 0,
            "zero_crossing_rate": 0
        }
    
    try:
        # Extract features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        energy = np.mean(librosa.feature.rms(y=audio_data)[0])
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0])
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data)[0])
        
        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy": energy,
            "tempo": tempo,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return {
            "pitch_mean": 0,
            "pitch_std": 0,
            "energy": 0,
            "tempo": 0,
            "spectral_centroid": 0,
            "spectral_bandwidth": 0,
            "spectral_rolloff": 0,
            "zero_crossing_rate": 0
        }


# Function to analyze facial features and emotions
def analyze_face(frame):
    try:
        # Use DeepFace for emotion analysis
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['emotion', 'age', 'gender'],
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
        
        # Calculate valence and arousal as weighted average of all emotions
        valence = 0.0
        arousal = 0.0
        total_weight = 0.0
        
        for emotion, score in emotions.items():
            if emotion in emotion_mapping:
                valence += emotion_mapping[emotion]['valence'] * score / 100
                arousal += emotion_mapping[emotion]['arousal'] * score / 100
                total_weight += score / 100
        
        if total_weight > 0:
            valence /= total_weight
            arousal /= total_weight
        else:
            valence = 0.5
            arousal = 0.5
        
        # Extract face landmarks
        landmarks = extract_face_landmarks(frame)
        
        # Calculate eye aspect ratio for cognitive load estimation
        ear = calculate_ear(landmarks, frame)
        cognitive_load = max(0, min(1, 1.0 - ear * 2))  # Inverse relationship with EAR
        
        return {
            'valence': valence,
            'arousal': arousal,
            'emotions': emotions,
            'ear': ear,
            'cognitive_load': cognitive_load,
            'landmarks': landmarks
        }
    except Exception as e:
        print(f"Error in face analysis: {e}")
        return None


# Function to calculate all metrics based on face and audio analysis
def calculate_metrics(face_data, audio_features):
    if not face_data:
        # Return default values if no face is detected
        return {metric: 0.5 for metric in metrics}
    
    # Extract basic data from face analysis
    valence = face_data['valence']
    arousal = face_data['arousal']
    cognitive_load = face_data['cognitive_load']
    
    # Calculate dominance (using a simplified model)
    # Higher dominance correlates with higher arousal and higher valence
    dominance = (valence * 0.6) + (arousal * 0.4)
    
    # Calculate personality traits (simplified model)
    # These would normally require more complex analysis or questionnaires
    
    # Extraversion - related to expressiveness, higher arousal
    extraversion = (arousal * 0.7) + (valence * 0.3)
    
    # Neuroticism - inverse relationship with emotional stability
    # Higher cognitive load and higher arousal with lower valence indicates higher neuroticism
    neuroticism = (cognitive_load * 0.3) + (arousal * 0.3) + ((1 - valence) * 0.4)
    neuroticism = max(0, min(1, neuroticism))
    
    # Emotional stability (inverse of neuroticism)
    emotional_stability = 1 - neuroticism
    
    # Openness - more varied expressions and voice patterns
    if audio_features:
        pitch_variability = min(1, audio_features['pitch_std'] / 100)
        openness = (pitch_variability * 0.5) + (0.5 * (1 - abs(0.5 - valence)))
    else:
        openness = 0.5 + (0.5 * (1 - abs(0.5 - valence)))
    
    # Agreeableness - higher with positive valence
    agreeableness = (valence * 0.7) + ((1 - arousal) * 0.3)
    
    # Conscientiousness - steady patterns, moderate arousal
    conscientiousness = (1 - abs(arousal - 0.5)) * 0.7 + (emotional_stability * 0.3)
    
    # Stress index - combination of cognitive load, high arousal, low valence
    stress_index = (cognitive_load * 0.4) + (arousal * 0.3) + ((1 - valence) * 0.3)
    stress_index = max(0, min(1, stress_index))
    
    # Engagement level - combination of arousal and cognitive activity
    engagement_level = (arousal * 0.5) + ((1 - cognitive_load) * 0.5)
    
    return {
        'valence': valence,
        'arousal': arousal,
        'dominance': dominance,
        'cognitive_load': cognitive_load,
        'emotional_stability': emotional_stability,
        'openness': openness,
        'agreeableness': agreeableness,
        'neuroticism': neuroticism,
        'conscientiousness': conscientiousness,
        'extraversion': extraversion,
        'stress_index': stress_index,
        'engagement_level': engagement_level
    }


# Function to update the metrics display
def update_metrics_display(metrics_data):
    # Create two rows of gauge charts
    fig = make_subplots(
        rows=3, 
        cols=4,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Define the metrics layout
    metric_layout = [
        # Row 1
        {'name': 'valence', 'row': 1, 'col': 1, 'title': 'Valence', 'color': 'blue'},
        {'name': 'arousal', 'row': 1, 'col': 2, 'title': 'Arousal', 'color': 'red'},
        {'name': 'dominance', 'row': 1, 'col': 3, 'title': 'Dominance', 'color': 'purple'},
        {'name': 'cognitive_load', 'row': 1, 'col': 4, 'title': 'Cognitive Load', 'color': 'orange'},
        
        # Row 2
        {'name': 'emotional_stability', 'row': 2, 'col': 1, 'title': 'Emotional Stability', 'color': 'green'},
        {'name': 'openness', 'row': 2, 'col': 2, 'title': 'Openness', 'color': 'teal'},
        {'name': 'agreeableness', 'row': 2, 'col': 3, 'title': 'Agreeableness', 'color': 'pink'},
        {'name': 'neuroticism', 'row': 2, 'col': 4, 'title': 'Neuroticism', 'color': 'brown'},
        
        # Row 3
        {'name': 'conscientiousness', 'row': 3, 'col': 1, 'title': 'Conscientiousness', 'color': 'darkblue'},
        {'name': 'extraversion', 'row': 3, 'col': 2, 'title': 'Extraversion', 'color': 'gold'},
        {'name': 'stress_index', 'row': 3, 'col': 3, 'title': 'Stress Index', 'color': 'crimson'},
        {'name': 'engagement_level', 'row': 3, 'col': 4, 'title': 'Engagement', 'color': 'lime'}
    ]
    
    # Add each gauge chart
    for metric in metric_layout:
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics_data[metric['name']],
                title={'text': metric['title']},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': metric['color']},
                    'steps': [
                        {'range': [0, 0.33], 'color': 'lightgray'},
                        {'range': [0.33, 0.66], 'color': 'gray'},
                        {'range': [0.66, 1], 'color': 'darkgray'}
                    ]
                }
            ),
            row=metric['row'],
            col=metric['col']
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=30, b=30),
        font=dict(size=10)
    )
    
    return fig


# Function to update history charts
def update_history_charts(df):
    if len(df) < 2:
        # Not enough data points
        fig = go.Figure()
        fig.update_layout(title="Waiting for more data points...")
        return fig
    
    # Create time series plot for metrics over time
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                     subplot_titles=("Emotional Dimensions", "Personality Traits", "Cognitive Measures"))
    
    # Row 1: Emotional dimensions
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['valence'], mode='lines+markers', name='Valence', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['arousal'], mode='lines+markers', name='Arousal', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['dominance'], mode='lines+markers', name='Dominance', line=dict(color='purple')), row=1, col=1)
    
    # Row 2: Personality traits
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['openness'], mode='lines+markers', name='Openness', line=dict(color='teal')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['agreeableness'], mode='lines+markers', name='Agreeableness', line=dict(color='pink')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['neuroticism'], mode='lines+markers', name='Neuroticism', line=dict(color='brown')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['conscientiousness'], mode='lines+markers', name='Conscientiousness', line=dict(color='darkblue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['extraversion'], mode='lines+markers', name='Extraversion', line=dict(color='gold')), row=2, col=1)
    
    # Row 3: Cognitive measures
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cognitive_load'], mode='lines+markers', name='Cognitive Load', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['emotional_stability'], mode='lines+markers', name='Emotional Stability', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stress_index'], mode='lines+markers', name='Stress Index', line=dict(color='crimson')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['engagement_level'], mode='lines+markers', name='Engagement', line=dict(color='lime')), row=3, col=1)
    
    # Update layout
    fig.update_layout(height=700, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    
    return fig


# Handle button clicks
if start_button:
    start_audio_recording()

if stop_button:
    stop_audio_recording()


# Main analysis loop
try:
    # Initialize camera
    cap = cv2.VideoCapture(selected_camera)
    
    last_analysis_time = time.time()
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture video. Please check camera settings.")
            break
        
        # Display camera feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze face on regular intervals
        current_time = time.time()
        elapsed_time = current_time - st.session_state.start_time
        
        # Perform analysis at the specified frequency
        if current_time - last_analysis_time >= analysis_freq:
            # Analyze face
            face_data = analyze_face(frame)
            st.session_state.face_data = face_data
            
            # Extract audio features from recent audio
            audio_features = None
            if len(st.session_state.audio_data) > 0:
                audio_features = extract_audio_features(st.session_state.audio_data[-int(sample_rate*2):], sample_rate)
            
            # Calculate metrics
            calculated_metrics = calculate_metrics(face_data, audio_features)
            
            # Add timestamp
            current_metrics = {
                'timestamp': elapsed_time,
                **calculated_metrics
            }
            
            # Update dataframe
            new_row = pd.DataFrame([current_metrics])
            st.session_state.metrics_data = pd.concat([st.session_state.metrics_data, new_row], ignore_index=True)
            
            last_analysis_time = current_time
        
        # Draw face mesh if landmarks are available
        if st.session_state.face_data and st.session_state.face_data['landmarks']:
            mesh_frame = frame_rgb.copy()
            landmarks = st.session_state.face_data['landmarks']
            mp_drawing.draw_landmarks(
                image=mesh_frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            camera_placeholder.image(mesh_frame, channels="RGB", use_column_width=True)
        else:
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update metrics display if we have data
        if not st.session_state.metrics_data.empty:
            # Get the most recent metrics
            latest_metrics = st.session_state.metrics_data.iloc[-1].to_dict()
            
            # Update gauge charts
            metrics_fig = update_metrics_display(latest_metrics)
            metrics_placeholder.plotly_chart(metrics_fig, use_container_width=True)
            
            # Update history charts
            history_fig = update_history_charts(st.session_state.metrics_data)
            history_placeholder.plotly_chart(history_fig, use_container_width=True)
            
            # Update data table
            data_placeholder.dataframe(st.session_state.metrics_data.round(2), use_container_width=True)
        
        # Sleep to control refresh rate
        time.sleep(0.05)

except Exception as e:
    st.error(f"An error occurred: {e}")
finally:
    # Clean up resources
    if 'cap' in locals() and cap is not None:
        cap.release()
