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
    st.session_state['run_analysis'] = False # Still useful to control UI elements
if 'campaign_description' not in st.session_state:
    st.session_state['campaign_description'] = ""

# --- WebRTC Callback Class (Defined but NOT USED in this version) ---
# Keep the definition here for easy re-enabling later
class AffectiveAIDebugProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        print(">>> Processor Initialized (But not used in current config)") # Log initialization
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        # print(f">>> Received video frame {self.frame_count} at {time.time():.2f}") # Log frame reception

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
st.title("👁️🎙️ Real-Time Affective AI Demo (Debug Mode 2)")
st.markdown("""
    **Debugging Mode 2:** Further simplified to test basic webcam streaming stability.
    **Custom video processing is disabled.** We are only trying to display the raw webcam feed directly.

    **Instructions:**
    1. Enter any text in the Campaign Context box.
    2. Click "Start Stream".
    3. Grant camera permissions if prompted.
    4. Observe if the raw video stream appears and stays active.
    5. **Check Browser Console (F12 -> Console) for errors if it fails.**
    6. Check Hugging Face logs for errors.
""")

# --- Campaign Input ---
st.subheader("Campaign Context")
campaign_input = st.text_area(
    "Enter the Campaign Description or Content:",
    value=st.session_state.get('campaign_description', "Debug Test 2"),
    height=100,
    key="campaign_description_input"
)
st.session_state['campaign_description'] = campaign_input


# --- Controls & WebRTC ---
st.subheader("Stream Control")
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    # Changed button label for clarity
    if st.button("Start Stream", key="start_button_debug2"):
        st.session_state['run_analysis'] = True # Keep state for UI consistency
        st.info("Attempting to start raw video stream...")
        print("--- Start Stream Button Clicked ---")

with col_ctrl2:
     # Changed button label for clarity
    if st.button("Stop Stream", key="stop_button_debug2"):
        st.session_state['run_analysis'] = False
        st.info("Stream stopped.")
        print("--- Stop Stream Button Clicked ---")


# --- WebRTC Streamer (Highly Simplified Call) ---
# Define error handler callback (Defined but NOT USED in this version)
def handle_error(error):
    print(f"!!! WebRTC Component Error: {error}")
    st.error(f"WebRTC Error: {error}") # Display error in UI

# Streamer component - Removed processor and error handler
try:
    print("--- Calling webrtc_streamer ---")
    ctx = webrtc_streamer(
        key="affective-ai-debug-noproc", # Use a new distinct key
        mode=WebRtcMode.SENDRECV, # Still send/recv to display feed
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={"video": True, "audio": False}, # Video only
        video_processor_factory=None, # *** REMOVED PROCESSOR ***
        audio_processor_factory=None,
        async_processing=True,
        # on_error=handle_error # *** REMOVED ERROR HANDLER ***
    )
    print("--- webrtc_streamer call finished ---")
except Exception as e:
    # Catch potential errors during the component call itself
    print(f"!!! ERROR during webrtc_streamer call: {type(e).__name__}: {e}")
    st.error(f"Error initializing WebRTC component: {e}")
    ctx = None # Ensure ctx is None if call failed

# --- Status Display ---
st.subheader("Stream Status")
if ctx and ctx.state.playing:
    st.success("WebRTC Status: PLAYING (Raw stream should be active)")
elif ctx:
    st.error(f"WebRTC Status: {ctx.state.name} (Stream is not active)")
else:
    st.error("WebRTC Status: Component failed to initialize.")


st.markdown("---")
st.markdown("Debug Version 2")
```

**Key Changes in this Version:**

1.  **`webrtc_streamer` Call:**
    * `video_processor_factory` is explicitly set to `None`.
    * `on_error` argument is removed.
    * A new `key` is used (`affective-ai-debug-noproc`) to prevent state conflicts.
    * Added `print` statements before and after the call.
    * Wrapped the call in a `try...except` block to catch immediate errors during initialization.
2.  **Processor Class:** The `AffectiveAIDebugProcessor` class definition remains, but it's no longer passed to `webrtc_streamer`.
3.  **UI Text:** Updated button labels and descriptions to reflect that this version attempts to show the *raw* stream without processing.

**Please try this version.**

* **If it works** (you see your raw webcam feed): The `TypeError` is related to the `AffectiveAIDebugProcessor` class or the `handle_error` function. The next step would be to re-introduce the processor *without* inheriting from `VideoProcessorBase` or simplify the `handle_error` function.
* **If it still fails with the same `TypeError` at the same line:** This is more puzzling. It might indicate a deeper issue within `streamlit-webrtc` or its interaction with other libraries/environment on Hugging Face Spaces, possibly related to the other arguments like `rtc_configuration` or `media_stream_constraints`. Check the Hugging Face logs and browser console very carefully for any clu
