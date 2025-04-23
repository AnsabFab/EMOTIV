import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import cv2
import numpy as np
# import mediapipe as mp # Commented out for debugging
import av # Required by streamlit-webrtc
# import librosa # Commented out for debugging
import pandas as pd
import time
import threading
from collections import deque
# import queue # Commented out for debugging
import os

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Affective AI Demo (Debug)")
print("--- Streamlit App Reload ---") # Log app reloads

# --- State Management (Minimal for Debugging) ---
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False
if 'campaign_description' not in st.session_state:
    st.session_state['campaign_description'] = ""

# --- WebRTC Callback Class (Extremely Simplified) ---
# Inherit from VideoProcessorBase for clarity
class AffectiveAIDebugProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        print(">>> Processor Initialized") # Log initialization
        self.frame_count = 0

    # The recv method is called for each frame
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        print(f">>> Received video frame {self.frame_count} at {time.time():.2f}") # Log frame reception

        # Check if analysis is stopped in session state
        # Note: Accessing session state directly from background thread might be tricky/unsafe
        # Rely on the component stopping sending frames when stopped.

        try:
            img = frame.to_ndarray(format="bgr24")
            # Minimal processing: Add a timestamp and frame count
            timestamp = time.strftime("%H:%M:%S")
            text = f"{timestamp} | Frame: {self.frame_count}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # print(f">>> Processed video frame {self.frame_count} at {time.time():.2f}") # Log successful processing (can be verbose)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"!!! ERROR processing video frame {self.frame_count}: {type(e).__name__}: {e}")
            # Return the original frame in case of error to try keep stream alive
            return frame

# --- Streamlit UI ---
st.title("👁️🎙️ Real-Time Affective AI Demo (Debug Mode)")
st.markdown("""
    **Debugging Mode:** This version is highly simplified to test basic webcam streaming stability.
    MediaPipe analysis, audio processing, and metrics calculation are disabled.
    We are only trying to echo the video feed with a timestamp.

    **Instructions:**
    1. Enter any text in the Campaign Context box.
    2. Click "Start Analysis".
    3. Grant camera permissions if prompted.
    4. Observe if the video stream stays active.
    5. **Check Browser Console (F12 -> Console) for errors if it fails.**
    6. Check Hugging Face logs for errors.
""")

# --- Campaign Input ---
st.subheader("Campaign Context")
campaign_input = st.text_area(
    "Enter the Campaign Description or Content:",
    value=st.session_state.get('campaign_description', "Debug Test"),
    height=100,
    key="campaign_description_input"
)
st.session_state['campaign_description'] = campaign_input


# --- Controls & WebRTC ---
st.subheader("Real-Time Analysis Control")
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    if st.button("Start Analysis", key="start_button_debug"):
        st.session_state['run_analysis'] = True
        st.info("Attempting to start video stream...")
        print("--- Start Analysis Button Clicked ---")

with col_ctrl2:
    if st.button("Stop Analysis", key="stop_button_debug"):
        st.session_state['run_analysis'] = False
        st.info("Analysis stopped.")
        print("--- Stop Analysis Button Clicked ---")


# --- WebRTC Streamer (Simplified Call) ---
# Define error handler callback
def handle_error(error):
    print(f"!!! WebRTC Component Error: {error}")
    st.error(f"WebRTC Error: {error}") # Display error in UI

# Streamer component
ctx = webrtc_streamer(
    key="affective-ai-debug", # Use a distinct key for debugging
    mode=WebRtcMode.SENDRECV, # Send local stream, receive processed stream
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Standard Google STUN
    }),
    # Request only video, disable audio entirely for this test
    media_stream_constraints={"video": True, "audio": False},
    # Use the simplified video-only processor factory
    video_processor_factory=AffectiveAIDebugProcessor,
    # Explicitly set audio processor factory to None
    audio_processor_factory=None,
    async_processing=True, # Still recommended
    # Add error handler
    on_error=handle_error
)

# --- Status Display ---
st.subheader("Stream Status")
if ctx.state.playing:
    st.success("WebRTC Status: PLAYING (Stream should be active)")
    if st.session_state.get('run_analysis', False):
         st.info("Simplified video processing is active.")
    else:
         st.warning("Analysis is stopped via button, but component might still be technically playing.")
else:
    st.error(f"WebRTC Status: {ctx.state.name} (Stream is not active)")


st.markdown("---")
st.markdown("Debug Version")

