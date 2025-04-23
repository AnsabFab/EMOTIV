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

# MediaPipe Setup (Keep setup for later re-enabling)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- State Management ---
# Initialize session state variables if they don't exist
if 'metrics_log' not in st.session_state:
    st.session_state['metrics_log'] = pd.DataFrame(columns=[
        "Timestamp", "Campaign_Description", "Valence", "Arousal", "Engagement_Proxy",
        "Stress_Proxy", "Cognitive_Load_Proxy", "Blink_Detected",
        "Avg_Pupil_Proxy", "Audio_RMS"
    ])
if 'latest_metrics' not in st.session_state:
     st.session_state['latest_metrics'] = {}
if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = deque(maxlen=20)
if 'video_buffer' not in st.session_state:
     st.session_state['video_buffer'] = deque(maxlen=10)
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False
if 'campaign_description' not in st.session_state:
    st.session_state['campaign_description'] = ""
if 'last_log_time' not in st.session_state:
    st.session_state['last_log_time'] = time.time()

# Thread lock for safe access to shared state
lock = threading.Lock()

# --- Helper Functions (Keep definitions for later re-enabling) ---
# These functions provide proxy calculations and are kept for when DEBUG_MODE is False
def map_emotion_proxy_to_va(landmarks):
    valence = 0.5; arousal = 0.5
    try:
        # Example: Check mouth corners (landmarks 61, 291) relative to nose (landmark 1)
        left_corner_y = landmarks.landmark[61].y
        right_corner_y = landmarks.landmark[291].y
        nose_y = landmarks.landmark[1].y
        if (left_corner_y < nose_y) and (right_corner_y < nose_y):
            valence = 0.8; arousal = 0.6 # Smile proxy

        # Example: Check eyebrow height (landmarks 52, 282) relative to eyes
        left_eyebrow_y = landmarks.landmark[52].y
        right_eyebrow_y = landmarks.landmark[282].y
        left_eye_y = landmarks.landmark[145].y
        right_eye_y = landmarks.landmark[374].y
        if (left_eyebrow_y < left_eye_y - 0.02) and (right_eyebrow_y < right_eye_y - 0.02):
             arousal = max(arousal, 0.75) # Surprise/fear proxy
    except (IndexError, AttributeError): pass # Ignore if landmarks aren't available
    return valence, arousal

def calculate_pupil_proxy(landmarks, frame_shape):
    try:
        # Using iris landmarks (indices from MediaPipe documentation)
        l_center = landmarks.landmark[473]; l_top = landmarks.landmark[474]; l_bottom = landmarks.landmark[475]
        l_v_dist = np.sqrt((l_top.x - l_bottom.x)**2 + (l_top.y - l_bottom.y)**2)
        r_center = landmarks.landmark[468]; r_top = landmarks.landmark[469]; r_bottom = landmarks.landmark[470]
        r_v_dist = np.sqrt((r_top.x - r_bottom.x)**2 + (r_top.y - r_bottom.y)**2)
        # Average vertical distance relative to frame height
        avg_iris_diam_relative = ((l_v_dist + r_v_dist) / 2)
        # Crude normalization - needs calibration
        pupil_proxy = min(avg_iris_diam_relative * 30, 1.0) # Adjust multiplier based on typical values
        return max(0.0, pupil_proxy)
    except (IndexError, AttributeError, ZeroDivisionError): return 0.5 # Default on error

def detect_blink(landmarks):
    try:
        # Standard eye aspect ratio calculation
        def eye_aspect_ratio(eye_landmarks_indices):
            pts = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_landmarks_indices], dtype=np.float32)
            v1 = np.linalg.norm(pts[1] - pts[5]); v2 = np.linalg.norm(pts[2] - pts[4])
            h = np.linalg.norm(pts[0] - pts[3])
            if h < 1e-6: return 0.3 # Avoid division by zero, return open state
            ear = (v1 + v2) / (2.0 * h)
            return ear
        left_ear = eye_aspect_ratio([33, 160, 158, 133, 153, 144])
        right_ear = eye_aspect_ratio([362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0
        BLINK_THRESHOLD = 0.20 # Adjust based on testing
        return avg_ear < BLINK_THRESHOLD
    except (IndexError, AttributeError): return False # Default on error

# --- WebRTC Callback Class ---
class AffectiveAIProcessor:
    def __init__(self) -> None:
        # No instance state needed for simplified version
        pass

    def process_video_simplified(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Simplified: Just return the frame with a timestamp overlay
        img = frame.to_ndarray(format="bgr24")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def process_video_full(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Full processing logic with MediaPipe
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False # Improve performance
        results = face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB_BGR) # Convert back for drawing

        # Default values
        face_detected = False
        valence_proxy = 0.5
        arousal_proxy = 0.5
        pupil_proxy = 0.5
        blink_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks: # Should be only one due to max_num_faces=1
                # Draw annotations
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=1))
                mp_drawing.draw_landmarks(image=img_bgr, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

                # Calculate features using helper functions
                valence_proxy, arousal_proxy = map_emotion_proxy_to_va(face_landmarks)
                pupil_proxy = calculate_pupil_proxy(face_landmarks, img_bgr.shape)
                blink_detected = detect_blink(face_landmarks)

                # Store features in session state buffer (use lock for thread safety)
                with lock:
                    st.session_state['video_buffer'].append({
                        "valence": valence_proxy, "arousal": arousal_proxy, "pupil": pupil_proxy,
                        "blink": blink_detected, "detected": True, "timestamp": time.time()
                    })
                break # Process only the first detected face
        else:
             # No face detected, store default values
             with lock:
                st.session_state['video_buffer'].append({
                    "valence": 0.5, "arousal": 0.5, "pupil": 0.5,
                    "blink": False, "detected": False, "timestamp": time.time()
                })

        # Return the annotated frame
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    def process_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Simplified: Don't process audio during debugging to isolate video issues
        # You can re-enable this later if the video stream is stable
        # try:
        #     raw_samples = frame.to_ndarray()
        #     # Ensure mono float32 samples
        #     mono_samples = raw_samples.mean(axis=0).astype(np.float32) if raw_samples.ndim > 1 else raw_samples.astype(np.float32)
        #     if mono_samples.size > 0:
        #         with lock: # Use lock for thread safety
        #             st.session_state['audio_buffer'].append(mono_samples)
        # except Exception as e:
        #     # print(f"Error processing audio frame: {e}") # Optional: log error
        #     pass
        return frame # Echo audio back without processing

    def recv(self, frame: av.AudioFrame | av.VideoFrame) -> av.AudioFrame | av.VideoFrame:
        # --- Control processing mode ---
        # Set DEBUG_MODE to False to re-enable full MediaPipe analysis
        DEBUG_MODE = True

        # Check if analysis should run
        if not st.session_state.get('run_analysis', False):
             return frame # Pass frame through if analysis is stopped

        try: # Add top-level try-except in callback for robustness
            if isinstance(frame, av.VideoFrame):
                if DEBUG_MODE:
                    # Call simplified processing (just timestamp overlay)
                    return self.process_video_simplified(frame)
                else:
                    # Call full processing with MediaPipe
                    return self.process_video_full(frame)
            elif isinstance(frame, av.AudioFrame):
                # Pass audio through without processing in either mode for now
                # return self.process_audio(frame) # Re-enable if needed
                return frame
            else:
                # Pass through unknown frame types
                return frame
        except Exception as e:
            # Log any error occurring during processing to Streamlit console
            print(f"Error in recv callback: {type(e).__name__}: {e}")
            # Optionally display error in UI (might clutter)
            # st.error(f"Processing Error: {e}")
            # Return the original frame to try and keep the stream alive
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
    value=st.session_state.get('campaign_description', "Default Campaign Context - Please Replace"),
    height=150,
    key="campaign_description_input" # Assign key for state management
)
# Update session state when text area changes
st.session_state['campaign_description'] = campaign_input


# --- Controls & WebRTC ---
st.subheader("Real-Time Analysis")
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    # Start Button Logic
    if st.button("Start Analysis", key="start_button"):
        if not st.session_state['campaign_description']:
             st.warning("Please enter a campaign description before starting.")
        else:
            st.session_state['run_analysis'] = True
            # Clear previous data on start
            st.session_state['video_buffer'].clear()
            st.session_state['audio_buffer'].clear()
            # Reset DataFrame to empty with correct columns
            st.session_state['metrics_log'] = pd.DataFrame(columns=[
                "Timestamp", "Campaign_Description", "Valence", "Arousal", "Engagement_Proxy",
                "Stress_Proxy", "Cognitive_Load_Proxy", "Blink_Detected",
                "Avg_Pupil_Proxy", "Audio_RMS"
            ])
            st.session_state['last_log_time'] = time.time() # Reset log timer
            st.info("Analysis started. Webcam/Mic should activate.")

with col_ctrl2:
    # Stop Button Logic
    if st.button("Stop Analysis", key="stop_button"):
        st.session_state['run_analysis'] = False
        st.info("Analysis stopped.")

# WebRTC Streamer Component
ctx = webrtc_streamer(
    key="affective-ai",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), # Use Google's public STUN server
    media_stream_constraints={"video": True, "audio": True},
    video_processor_factory=AffectiveAIProcessor, # Use our class for video
    audio_processor_factory=AffectiveAIProcessor, # Use our class for audio (even if simplified)
    async_processing=True, # Process frames in background threads
)

# --- Display Metrics (Conditional based on DEBUG_MODE concept) ---
st.subheader("Estimated Metrics (Real-Time Proxies)")
placeholder = st.empty() # Use a placeholder to update metrics display smoothly

# Check if the stream is active and analysis is supposed to be running
if ctx.state.playing and st.session_state.get('run_analysis', False):

    # --- Check if we are in simplified mode (based on the flag in AffectiveAIProcessor) ---
    # Since we can't directly access the instance's DEBUG_MODE flag here easily,
    # we'll rely on the fact that the video_buffer will likely be empty or have defaults
    # if full processing isn't happening. A cleaner way might involve passing state back via queues.

    # For this version, let's assume if DEBUG_MODE is True in the processor, we show the info message.
    # If DEBUG_MODE is False, we proceed with metrics calculation and display.
    # We'll control this by commenting/uncommenting the metrics block below when changing DEBUG_MODE.

    # --- Placeholder message during simplified debugging ---
    # If DEBUG_MODE = True in AffectiveAIProcessor, the code below this comment block
    # should ideally be commented out or conditionally skipped.
    placeholder.info("Running in simplified mode. Video stream should be active if connection is stable. Metrics calculation disabled.")

    # --- Original Metrics Display & Logging Logic ---
    # UNCOMMENT THE BLOCK BELOW WHEN SETTING DEBUG_MODE = False in AffectiveAIProcessor
    """
    with placeholder.container(): # Update metrics within the container
        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics from buffers (use lock for thread safety)
        with lock:
            recent_video = list(st.session_state['video_buffer'])
            recent_audio = list(st.session_state['audio_buffer'])

        # --- Calculate average video features ---
        if recent_video:
            avg_valence = np.mean([f['valence'] for f in recent_video])
            avg_arousal = np.mean([f['arousal'] for f in recent_video])
            avg_pupil = np.mean([f['pupil'] for f in recent_video])
            face_detect_ratio = np.mean([f['detected'] for f in recent_video])
            # Use the blink status from the most recent frame in the buffer
            last_blink = recent_video[-1]['blink'] if recent_video else False
            engagement = face_detect_ratio # Simple engagement proxy
            stress = max(0, (1.0 - avg_valence) * avg_arousal) # Simple stress proxy
            cog_load = (avg_pupil + avg_arousal) / 2.0 # Simple cog load proxy
        else:
            # Default values if buffer is empty
            avg_valence, avg_arousal, engagement, stress, cog_load, last_blink, avg_pupil = 0.5, 0.5, 0.0, 0.0, 0.5, False, 0.5

        # --- Calculate audio features ---
        norm_audio_rms = 0.0
        if recent_audio:
             try:
                # Concatenate recent audio chunks
                concatenated_audio = np.concatenate(recent_audio)
                if concatenated_audio.size > 0:
                    # Calculate Root Mean Square (proxy for volume/energy)
                    audio_rms = np.sqrt(np.mean(concatenated_audio**2))
                    # Normalize RMS somewhat (highly dependent on mic gain/distance)
                    norm_audio_rms = min(audio_rms * 10, 1.0)
                    # Factor audio intensity into arousal (example weighting)
                    avg_arousal = (avg_arousal * 0.7) + (norm_audio_rms * 0.3)
             except ValueError: pass # Handle case where buffer might be temporarily empty during concat

        # Store latest calculated metrics for logging
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
        st.session_state['latest_metrics'] = latest_metrics_data # Update state

        # --- Display metrics using Streamlit columns ---
        col1.metric("Valence", f"{latest_metrics_data['Valence']:.2f}", f"{latest_metrics_data['Valence']-0.5:.2f}")
        col2.metric("Arousal", f"{latest_metrics_data['Arousal']:.2f}", f"{latest_metrics_data['Arousal']-0.5:.2f}")
        col3.metric("Engagement", f"{latest_metrics_data['Engagement_Proxy']:.2f}", f"{latest_metrics_data['Engagement_Proxy']-0.5:.2f}")
        col4.metric("Stress", f"{latest_metrics_data['Stress_Proxy']:.2f}", f"{latest_metrics_data['Stress_Proxy']-0.2:.2f}") # Delta relative to low stress
        col1.metric("Cognitive Load", f"{latest_metrics_data['Cognitive_Load_Proxy']:.2f}", f"{latest_metrics_data['Cognitive_Load_Proxy']-0.5:.2f}")
        col2.metric("Pupil Proxy", f"{latest_metrics_data['Avg_Pupil_Proxy']:.2f}", f"{latest_metrics_data['Avg_Pupil_Proxy']-0.5:.2f}")
        col3.metric("Audio RMS", f"{latest_metrics_data['Audio_RMS']:.2f}", f"{latest_metrics_data['Audio_RMS']-0.1:.2f}") # Delta relative to quiet
        col4.metric("Blink Detected", "Yes" if latest_metrics_data['Blink_Detected'] else "No")

        # --- Data Logging Logic ---
        current_time = time.time()
        # Log data approximately every 4 seconds
        if current_time - st.session_state.get('last_log_time', current_time) >= 4.0:
            new_log_entry = pd.DataFrame([latest_metrics_data])
            # Ensure columns match before concatenating
            if set(new_log_entry.columns) != set(st.session_state['metrics_log'].columns) and not st.session_state['metrics_log'].empty:
                 st.warning("Log columns mismatch, resetting log.")
                 st.session_state['metrics_log'] = new_log_entry # Start fresh if columns changed
            else:
                 # Append new entry to the log DataFrame in session state
                 st.session_state['metrics_log'] = pd.concat([st.session_state['metrics_log'], new_log_entry], ignore_index=True)

            st.session_state['last_log_time'] = current_time # Update last log time

        # --- Display Logged Data Table ---
        st.subheader("Logged Metrics History (Last 10 Entries)")
        st.dataframe(st.session_state['metrics_log'].tail(10), use_container_width=True) # Show recent logs

        # --- Download Button ---
        # Function to convert DataFrame to CSV, cached for performance
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Show download button only if there's data logged
        if not st.session_state['metrics_log'].empty:
            csv_data = convert_df_to_csv(st.session_state['metrics_log'])
            st.download_button(
                label="Download Full Metrics Log as CSV",
                data=csv_data,
                file_name='affective_metrics_log.csv',
                mime='text/csv',
                key='download_button' # Add key for stability
            )
        else:
            st.info("No metrics logged yet to download.")
    """

# Display status messages if not playing or analysis stopped
elif not ctx.state.playing:
    placeholder.info("WebRTC streamer is not running. Click 'Start Analysis' after entering campaign info.")
elif not st.session_state.get('run_analysis', False):
     placeholder.info("Analysis stopped. Click 'Start Analysis' to resume.")


st.markdown("---")
st.markdown("Developed as a demonstration. Use ethically and responsibly.")
```

I've removed the multi-line comment block that contained the problematic apostrophe. Please try running this updated version. It should resolve the `SyntaxError`.

Remember that this version is still in the simplified debug mode (`DEBUG_MODE = True` within the `AffectiveAIProcessor` class). If the stream runs stably now, the next step is to set `DEBUG_MODE = False` and uncomment the metrics display block to re-enable the full analys
