import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import av # Required by streamlit-webrtc
import librosa
import pandas as pd
import time
import threading
from collections import deque
import queue # Thread-safe queue
import os # For checking file existence if needed

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Affective AI Demo")

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# Use static_image_mode=False, refine_landmarks=True for video streams
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # Essential for iris/pupil
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- State Management ---
# Initialize session state variables if they don't exist
if 'metrics_log' not in st.session_state:
    st.session_state['metrics_log'] = pd.DataFrame(columns=[
        "Timestamp", "Campaign_Description", "Valence", "Arousal", "Engagement_Proxy", # Added Campaign_Description
        "Stress_Proxy", "Cognitive_Load_Proxy", "Blink_Detected",
        "Avg_Pupil_Proxy", "Audio_RMS"
    ])
if 'latest_metrics' not in st.session_state:
     st.session_state['latest_metrics'] = {}
if 'audio_buffer' not in st.session_state:
    # Store raw audio chunks for processing
    st.session_state['audio_buffer'] = deque(maxlen=20) # Store ~1 second of audio chunks
if 'video_buffer' not in st.session_state:
     # Store recent video features
     st.session_state['video_buffer'] = deque(maxlen=10) # Store features from last 10 frames
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False # Control analysis start/stop
if 'campaign_description' not in st.session_state:
    st.session_state['campaign_description'] = "" # Store campaign description
if 'last_log_time' not in st.session_state:
    st.session_state['last_log_time'] = time.time() # Initialize log timer

# Thread lock for safe access to shared state
lock = threading.Lock()

# --- Helper Functions (Placeholder - Needs Refinement) ---

# Simplified mapping (Placeholder - Needs refinement)
def map_emotion_proxy_to_va(landmarks):
    valence = 0.5 # Neutral default
    arousal = 0.5 # Neutral default
    try:
        left_corner_y = landmarks.landmark[61].y
        right_corner_y = landmarks.landmark[291].y
        nose_y = landmarks.landmark[1].y
        if (left_corner_y < nose_y) and (right_corner_y < nose_y):
            valence = 0.8
            arousal = 0.6
        left_eyebrow_y = landmarks.landmark[52].y
        right_eyebrow_y = landmarks.landmark[282].y
        left_eye_y = landmarks.landmark[145].y
        right_eye_y = landmarks.landmark[374].y
        if (left_eyebrow_y < left_eye_y - 0.02) and (right_eyebrow_y < right_eye_y - 0.02):
             arousal = max(arousal, 0.75)
    except (IndexError, AttributeError): pass
    return valence, arousal

# Calculate relative pupil size (proxy) - VERY sensitive
def calculate_pupil_proxy(landmarks, frame_shape):
    try:
        l_center = landmarks.landmark[473]; l_top = landmarks.landmark[474]; l_bottom = landmarks.landmark[475]
        l_v_dist = np.sqrt((l_top.x - l_bottom.x)**2 + (l_top.y - l_bottom.y)**2)
        r_center = landmarks.landmark[468]; r_top = landmarks.landmark[469]; r_bottom = landmarks.landmark[470]
        r_v_dist = np.sqrt((r_top.x - r_bottom.x)**2 + (r_top.y - r_bottom.y)**2)
        avg_iris_diam_pixels = ((l_v_dist + r_v_dist) / 2) * frame_shape[0]
        pupil_proxy = min(avg_iris_diam_pixels / 15.0, 1.0)
        return max(0.0, pupil_proxy)
    except (IndexError, AttributeError, ZeroDivisionError): return 0.5

# Detect blinks (simple thresholding)
def detect_blink(landmarks):
    try:
        def eye_aspect_ratio(eye_landmarks_indices):
            pts = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_landmarks_indices], dtype=np.float32)
            v1 = np.linalg.norm(pts[1] - pts[5]); v2 = np.linalg.norm(pts[2] - pts[4])
            h = np.linalg.norm(pts[0] - pts[3])
            if h == 0: return 0.3
            ear = (v1 + v2) / (2.0 * h)
            return ear
        left_ear = eye_aspect_ratio([33, 160, 158, 133, 153, 144])
        right_ear = eye_aspect_ratio([362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0
        BLINK_THRESHOLD = 0.20
        return avg_ear < BLINK_THRESHOLD
    except (IndexError, AttributeError): return False

# --- WebRTC Callback Class ---
class AffectiveAIProcessor:
    def __init__(self) -> None:
        # Note: Using session_state for log timer now, so not needed here
        pass

    def process_video(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB_BGR)

        face_detected = False
        valence_proxy = 0.5
        arousal_proxy = 0.5
        pupil_proxy = 0.5
        blink_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh annotations
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1))
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

                # Calculate Features
                valence_proxy, arousal_proxy = map_emotion_proxy_to_va(face_landmarks)
                pupil_proxy = calculate_pupil_proxy(face_landmarks, img_bgr.shape)
                blink_detected = detect_blink(face_landmarks)

                # Store recent features
                with lock:
                    st.session_state['video_buffer'].append({
                        "valence": valence_proxy, "arousal": arousal_proxy, "pupil": pupil_proxy,
                        "blink": blink_detected, "detected": True, "timestamp": time.time()
                    })
                break # Process only first face
        else:
             # No face detected
             with lock:
                st.session_state['video_buffer'].append({
                    "valence": 0.5, "arousal": 0.5, "pupil": 0.5,
                    "blink": False, "detected": False, "timestamp": time.time()
                })

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    def process_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            raw_samples = frame.to_ndarray()
            mono_samples = raw_samples.mean(axis=0).astype(np.float32) if raw_samples.ndim > 1 else raw_samples.astype(np.float32)
            if mono_samples.size > 0:
                with lock:
                    st.session_state['audio_buffer'].append(mono_samples)
        except Exception as e: pass # Ignore audio errors silently for now
        return frame # Echo audio

    def recv(self, frame: av.AudioFrame | av.VideoFrame) -> av.AudioFrame | av.VideoFrame:
        if not st.session_state.get('run_analysis', False):
             return frame # Pass through if analysis stopped

        if isinstance(frame, av.VideoFrame):
            return self.process_video(frame)
        elif isinstance(frame, av.AudioFrame):
            return self.process_audio(frame)
        else:
            return frame

# --- Streamlit UI ---
st.title("👁️🎙️ Real-Time Affective AI Demo")
st.markdown("""
    This demo uses your webcam and microphone to estimate affective states in real-time
    while you view a campaign description.
    It leverages **MediaPipe** for facial landmark and iris tracking, and **Librosa** for basic audio analysis.

    **Disclaimer:**
    - These metrics are *proxies* and estimations. Accuracy varies significantly.
    - **Personality traits are NOT measured.** Cognitive Load and Stress are rough indicators.
    - Ensure good lighting and face visibility. Performance depends on your device.
    - Grant camera/microphone permissions when prompted by the browser.
""")

# --- Campaign Input ---
st.subheader("Campaign Context")
campaign_input = st.text_area(
    "Enter the Campaign Description or Content:",
    value=st.session_state.get('campaign_description', "Default Campaign Context - Please Replace"), # Use session state value
    height=150,
    key="campaign_description_input" # Assign a key
)
# Update session state when text area changes
st.session_state['campaign_description'] = campaign_input


# --- Controls & WebRTC ---
st.subheader("Real-Time Analysis")
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    if st.button("Start Analysis", key="start_button"):
        if not st.session_state['campaign_description']:
             st.warning("Please enter a campaign description before starting.")
        else:
            st.session_state['run_analysis'] = True
            # Reset buffers and log when starting a new analysis session
            st.session_state['video_buffer'].clear()
            st.session_state['audio_buffer'].clear()
            st.session_state['metrics_log'] = st.session_state['metrics_log'][0:0] # Clear DataFrame
            st.session_state['last_log_time'] = time.time() # Reset log timer
            st.info("Analysis started. Webcam/Mic should activate.")

with col_ctrl2:
    if st.button("Stop Analysis", key="stop_button"):
        st.session_state['run_analysis'] = False
        st.info("Analysis stopped.")

# WebRTC Streamer
ctx = webrtc_streamer(
    key="affective-ai",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": True},
    video_processor_factory=AffectiveAIProcessor,
    audio_processor_factory=AffectiveAIProcessor,
    async_processing=True,
)

# --- Display Metrics ---
st.subheader("Estimated Metrics (Real-Time Proxies)")
placeholder = st.empty() # Create a placeholder for metrics display

if ctx.state.playing and st.session_state.get('run_analysis', False):
    with placeholder.container(): # Use the placeholder to update metrics in place
        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics from buffers
        with lock:
            recent_video = list(st.session_state['video_buffer'])
            recent_audio = list(st.session_state['audio_buffer'])

        # Calculate average video features
        if recent_video:
            avg_valence = np.mean([f['valence'] for f in recent_video])
            avg_arousal = np.mean([f['arousal'] for f in recent_video])
            avg_pupil = np.mean([f['pupil'] for f in recent_video])
            face_detect_ratio = np.mean([f['detected'] for f in recent_video])
            last_blink = recent_video[-1]['blink']
            engagement = face_detect_ratio
            stress = max(0, (1.0 - avg_valence) * avg_arousal)
            cog_load = (avg_pupil + avg_arousal) / 2.0
        else:
            avg_valence, avg_arousal, engagement, stress, cog_load, last_blink, avg_pupil = 0.5, 0.5, 0.0, 0.0, 0.5, False, 0.5

        # Calculate audio features
        norm_audio_rms = 0.0
        if recent_audio:
             try:
                concatenated_audio = np.concatenate(recent_audio)
                if concatenated_audio.size > 0:
                    audio_rms = np.sqrt(np.mean(concatenated_audio**2))
                    norm_audio_rms = min(audio_rms * 10, 1.0) # Normalize somewhat
                    avg_arousal = (avg_arousal * 0.7) + (norm_audio_rms * 0.3) # Factor into arousal
             except ValueError: pass # Handle empty buffer case

        # Store latest for logging
        latest_metrics_data = {
            "Timestamp": pd.Timestamp.now(),
            "Campaign_Description": st.session_state.get('campaign_description', "N/A"), # Get current description
            "Valence": round(avg_valence, 3),
            "Arousal": round(avg_arousal, 3),
            "Engagement_Proxy": round(engagement, 3),
            "Stress_Proxy": round(stress, 3),
            "Cognitive_Load_Proxy": round(cog_load, 3),
            "Blink_Detected": last_blink,
            "Avg_Pupil_Proxy": round(avg_pupil, 3),
            "Audio_RMS": round(norm_audio_rms, 3)
        }
        st.session_state['latest_metrics'] = latest_metrics_data

        # Display metrics
        col1.metric("Valence", f"{latest_metrics_data['Valence']:.2f}", f"{latest_metrics_data['Valence']-0.5:.2f}")
        col2.metric("Arousal", f"{latest_metrics_data['Arousal']:.2f}", f"{latest_metrics_data['Arousal']-0.5:.2f}")
        col3.metric("Engagement", f"{latest_metrics_data['Engagement_Proxy']:.2f}", f"{latest_metrics_data['Engagement_Proxy']-0.5:.2f}")
        col4.metric("Stress", f"{latest_metrics_data['Stress_Proxy']:.2f}", f"{latest_metrics_data['Stress_Proxy']-0.2:.2f}")
        col1.metric("Cognitive Load", f"{latest_metrics_data['Cognitive_Load_Proxy']:.2f}", f"{latest_metrics_data['Cognitive_Load_Proxy']-0.5:.2f}")
        col2.metric("Pupil Proxy", f"{latest_metrics_data['Avg_Pupil_Proxy']:.2f}", f"{latest_metrics_data['Avg_Pupil_Proxy']-0.5:.2f}")
        col3.metric("Audio RMS", f"{latest_metrics_data['Audio_RMS']:.2f}", f"{latest_metrics_data['Audio_RMS']-0.1:.2f}")
        col4.metric("Blink Detected", "Yes" if latest_metrics_data['Blink_Detected'] else "No")

        # --- Data Logging ---
        current_time = time.time()
        if current_time - st.session_state.get('last_log_time', current_time) >= 4.0: # Log approx every 4 secs
            new_log_entry = pd.DataFrame([latest_metrics_data])
            # Ensure columns match if DataFrame was somehow cleared with different columns
            if set(new_log_entry.columns) != set(st.session_state['metrics_log'].columns):
                 st.warning("Log columns mismatch, resetting log.")
                 st.session_state['metrics_log'] = new_log_entry # Start fresh
            else:
                 st.session_state['metrics_log'] = pd.concat([st.session_state['metrics_log'], new_log_entry], ignore_index=True)

            st.session_state['last_log_time'] = current_time # Update log time

        # --- Display Logged Data ---
        st.subheader("Logged Metrics History (Last 10 Entries)")
        st.dataframe(st.session_state['metrics_log'].tail(10), use_container_width=True)

        # Download Button
        @st.cache_data # Cache the conversion
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        if not st.session_state['metrics_log'].empty:
            csv_data = convert_df_to_csv(st.session_state['metrics_log'])
            st.download_button(
                label="Download Full Metrics Log as CSV",
                data=csv_data,
                file_name='affective_metrics_log.csv',
                mime='text/csv',
                key='download_button'
            )
        else:
            st.info("No metrics logged yet to download.")

elif not ctx.state.playing:
    placeholder.info("WebRTC streamer is not running. Click 'Start Analysis' after entering campaign info.")
elif not st.session_state.get('run_analysis', False):
     placeholder.info("Analysis stopped. Click 'Start Analysis' to resume.")


st.markdown("---")
st.markdown("Developed as a demonstration. Use ethically and responsibly.")
