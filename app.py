import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import threading # Needed for locking
import av # Needed for streamlit-webrtc frame handling
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# Set page config (Ensure this is the first Streamlit command)
st.set_page_config(
    page_title="WebRTC Facial Analysis",
    page_icon="😀",
    layout="wide"
)

# --- MediaPipe Initialization (Keep as is) ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Consider initializing inside the processor if needed, but global might be fine
# depending on resource sharing needs. For simplicity, keep global for now.
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Metrics Definition (Keep as is) ---
metrics = [
    "valence", "arousal", "dominance", "cognitive_load",
    "emotional_stability", "openness", "agreeableness",
    "neuroticism", "conscientiousness", "extraversion",
    "stress_index", "engagement_level"
]

# --- Session State Initialization ---
# We'll store results and control state here, accessed via locks
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = pd.DataFrame(columns=['timestamp'] + metrics)
# 'is_analyzing' flag will now be primarily controlled via the processor or callback context
# if 'is_analyzing' not in st.session_state:
#    st.session_state.is_analyzing = False # Let the component manage its running state
if 'latest_metrics' not in st.session_state:
    st.session_state.latest_metrics = None
if 'latest_landmarks' not in st.session_state:
    st.session_state.latest_landmarks = None
if 'analysis_active' not in st.session_state:
     st.session_state.analysis_active = False # Separate flag for enabling analysis calculation


# --- Analysis Functions (Keep exactly as you provided) ---
def extract_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Optimize processing
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

def calculate_ear(landmarks): # Keep as is
    if not landmarks: return 0.0
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    def get_landmark_coords(landmark_indices):
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in landmark_indices])
    left_eye_points = get_landmark_coords(LEFT_EYE)
    right_eye_points = get_landmark_coords(RIGHT_EYE)
    def eye_aspect_ratio(eye_points):
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks): # Keep as is
    if not landmarks: return 0.0
    MOUTH_OUTLINE = [61, 291, 39, 181, 0, 17, 269, 405]
    mouth_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in MOUTH_OUTLINE])
    height = np.mean([
        np.linalg.norm(mouth_points[1] - mouth_points[5]),
        np.linalg.norm(mouth_points[2] - mouth_points[6]),
        np.linalg.norm(mouth_points[3] - mouth_points[7])
    ])
    width = np.linalg.norm(mouth_points[0] - mouth_points[4])
    return height / width if width > 0 else 0.0

def calculate_eyebrow_position(landmarks): # Keep as is
    if not landmarks: return 0.0
    LEFT_EYEBROW = 107; RIGHT_EYEBROW = 336
    LEFT_EYE = 159; RIGHT_EYE = 386
    left_eyebrow_y = landmarks.landmark[LEFT_EYEBROW].y
    right_eyebrow_y = landmarks.landmark[RIGHT_EYEBROW].y
    left_eye_y = landmarks.landmark[LEFT_EYE].y
    right_eye_y = landmarks.landmark[RIGHT_EYE].y
    left_distance = left_eye_y - left_eyebrow_y
    right_distance = right_eye_y - right_eyebrow_y
    avg_distance = (left_distance + right_distance) / 2.0
    normalized = (avg_distance - 0.02) / 0.06 # Approximate normalization
    return max(0.0, min(1.0, normalized))

def estimate_head_pose(landmarks): # Keep as is
    if not landmarks: return 0.0, 0.0
    NOSE_TIP = 4; LEFT_EYE = 159; RIGHT_EYE = 386
    nose = np.array([landmarks.landmark[NOSE_TIP].x, landmarks.landmark[NOSE_TIP].y, landmarks.landmark[NOSE_TIP].z])
    left_eye = np.array([landmarks.landmark[LEFT_EYE].x, landmarks.landmark[LEFT_EYE].y, landmarks.landmark[LEFT_EYE].z])
    right_eye = np.array([landmarks.landmark[RIGHT_EYE].x, landmarks.landmark[RIGHT_EYE].y, landmarks.landmark[RIGHT_EYE].z])
    eye_level = (left_eye[1] + right_eye[1]) / 2.0
    vertical_tilt = nose[1] - eye_level
    horizontal_mid = (left_eye[0] + right_eye[0]) / 2.0
    horizontal_tilt = nose[0] - horizontal_mid
    vertical_tilt = max(-1.0, min(1.0, vertical_tilt * 10)) # Normalize approx
    horizontal_tilt = max(-1.0, min(1.0, horizontal_tilt * 10)) # Normalize approx
    return vertical_tilt, horizontal_tilt

def calculate_metrics(landmarks): # Keep as is
    if not landmarks:
        return {metric: 0.5 for metric in metrics} # Return default values
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    eyebrow_position = calculate_eyebrow_position(landmarks)
    vertical_tilt, horizontal_tilt = estimate_head_pose(landmarks)
    cognitive_load = max(0, min(1, 1.0 - ear * 2.5))
    valence = max(0, min(1, mar * 2.0 * (1.0 - eyebrow_position)))
    arousal = max(0, min(1, (mar + (1.0 - ear) + eyebrow_position) / 3.0))
    dominance = max(0, min(1, 0.5 + vertical_tilt))
    neuroticism = max(0, min(1, (cognitive_load * 0.6) + ((1.0 - valence) * 0.4)))
    emotional_stability = 1.0 - neuroticism
    extraversion = max(0, min(1, (arousal * 0.5) + (valence * 0.5)))
    openness = max(0, min(1, 0.5 + ((mar - 0.5) * 0.5)))
    agreeableness = max(0, min(1, (valence * 0.7) + ((1.0 - arousal) * 0.3)))
    conscientiousness = max(0, min(1, (1.0 - abs(arousal - 0.5)) * 0.7 + (emotional_stability * 0.3)))
    stress_index = max(0, min(1, (cognitive_load * 0.5) + (eyebrow_position * 0.3) + ((1.0 - valence) * 0.2)))
    engagement_level = max(0, min(1, (arousal * 0.7) + ((1.0 - abs(horizontal_tilt)) * 0.3)))
    return {
        'valence': valence, 'arousal': arousal, 'dominance': dominance,
        'cognitive_load': cognitive_load, 'emotional_stability': emotional_stability,
        'openness': openness, 'agreeableness': agreeableness, 'neuroticism': neuroticism,
        'conscientiousness': conscientiousness, 'extraversion': extraversion,
        'stress_index': stress_index, 'engagement_level': engagement_level
    }

# --- Visualization Function (Keep as is, but check input) ---
def update_metrics_visualization(metrics_values):
    # (Your existing function - ensure it handles None input gracefully if needed)
    if not metrics_values: # Handle case where metrics haven't been calculated yet
         fig, ax = plt.subplots()
         ax.text(0.5, 0.5, "Waiting for analysis...", ha='center', va='center')
         ax.axis('off')
         return fig

    num_metrics = len([k for k in metrics_values if k != 'timestamp'])
    nrows = (num_metrics + 2) // 3 # Calculate rows needed (3 columns)
    fig, axs = plt.subplots(nrows, 3, figsize=(10, nrows * 2.5)) # Adjust figsize
    axs = axs.flatten()

    colors = [(0.1, 0.1, 0.9), (0.9, 0.9, 0.1), (0.9, 0.1, 0.1)] # Blue to Yellow to Red
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    norm = plt.Normalize(0, 1)

    metric_idx = 0
    for key, value in metrics_values.items():
        if key == 'timestamp': continue

        ax = axs[metric_idx]
        ax.set_title(key.replace('_', ' ').title(), fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.5); ax.set_aspect('equal'); ax.axis('off')
        r = 0.4 # radius
        theta = np.linspace(np.pi, 0, 100) # Flipped for gauge direction
        x_bg = 0.5 + r * np.cos(theta); y_bg = 0.1 + r * np.sin(theta)
        ax.plot(x_bg, y_bg, 'k-', linewidth=3, alpha=0.2) # Background arc
        value_angle = np.pi * (1 - value) # Map value [0,1] to angle [pi, 0]
        value_theta = np.linspace(np.pi, value_angle, max(2,int(100*value))) # Ensure at least 2 points
        x_val = 0.5 + r * np.cos(value_theta); y_val = 0.1 + r * np.sin(value_theta)
        points = np.array([x_val, y_val]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_values = np.linspace(0, value, len(segments)) # Color based on value
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_values); lc.set_linewidth(5)
        ax.add_collection(lc)
        ax.text(0.5, 0.15, f"{value:.2f}", ha='center', va='center', fontsize=11,
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        metric_idx += 1

    # Hide unused subplots
    for i in range(metric_idx, len(axs)): axs[i].axis('off')
    plt.tight_layout(pad=0.5); return fig


# --- WebRTC Video Processor ---
# Use a lock for thread-safe access to shared state variables like metrics_data
lock = threading.Lock()

class FacialAnalysisProcessor(VideoProcessorBase):
    def __init__(self, analysis_freq: float):
        self.analysis_freq = analysis_freq
        self.last_analysis_time = time.time()
        # Store latest results within the processor instance, protected by the lock
        self._latest_landmarks = None
        self._latest_metrics = None
        self._new_data_rows = [] # Temporary list to hold rows created in this thread


    def _update_shared_state(self, landmarks, metrics, data_row):
         # Use session state for simplicity, protected by lock
        with lock:
            st.session_state.latest_landmarks = landmarks
            st.session_state.latest_metrics = metrics
            if data_row:
                 # Append directly to the session state dataframe if it exists
                 # Ensure thread-safe append/concat if modifying shared df directly
                 # Easier: just store the latest metrics, let main thread handle df
                 new_row_df = pd.DataFrame([data_row])
                 st.session_state.metrics_data = pd.concat([st.session_state.metrics_data, new_row_df], ignore_index=True)


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert av.VideoFrame to NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        annotated_image = img.copy()
        current_landmarks = None # Landmarks for this specific frame
        perform_analysis = False

        # Check if analysis should run based on frequency and the global flag
        current_time = time.time()
        if st.session_state.analysis_active and (current_time - self.last_analysis_time >= self.analysis_freq):
             perform_analysis = True
             self.last_analysis_time = current_time

        if perform_analysis:
            # --- Run analysis ---
            current_landmarks = extract_face_landmarks(img) # Use original img for analysis
            calculated_metrics = calculate_metrics(current_landmarks)

            data_row = None
            if current_landmarks: # Only log if face detected
                 elapsed_time = current_time - st.session_state.start_time
                 data_row = {'timestamp': elapsed_time, **calculated_metrics}

            # --- Update shared state (thread-safe) ---
            self._update_shared_state(current_landmarks, calculated_metrics, data_row)

        # --- Drawing ---
        # Always try to draw the latest available landmarks from session state (might be slightly delayed)
        landmarks_to_draw = None
        with lock: # Read the latest landmarks safely
            landmarks_to_draw = st.session_state.latest_landmarks

        if landmarks_to_draw:
            # Draw landmarks on the annotated_image
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=landmarks_to_draw,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=landmarks_to_draw,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Convert processed NumPy array back to av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


# --- Streamlit UI Layout ---
st.title("WebRTC Facial Analysis")
st.markdown("""
This application uses your webcam via WebRTC to analyze facial landmarks and estimate psychological metrics in real-time.
*Note: Estimations are for demonstration only.*
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Webcam Feed")

    # Analysis frequency slider
    analysis_freq_input = st.slider("Analysis Frequency (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="freq_slider")

    # Control Button for Analysis Calculation
    if st.button("Start/Stop Analysis Calculation"):
         st.session_state.analysis_active = not st.session_state.analysis_active
         if not st.session_state.analysis_active:
              # Optionally clear metrics when stopping analysis calculation
              with lock:
                   st.session_state.latest_metrics = None
                   st.session_state.latest_landmarks = None


    status_text = "Analysis Calculation Active" if st.session_state.analysis_active else "Analysis Calculation Paused"
    st.info(f"Status: {status_text}")


    # RTC Configuration (Optional - mainly for deployment)
    # If running locally, often doesn't need explicit STUN servers
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # --- webrtc_streamer ---
    # This component handles the webcam feed
    webrtc_ctx = webrtc_streamer(
        key="facial-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: FacialAnalysisProcessor(analysis_freq=analysis_freq_input),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Process frames in background thread
    )

    if not webrtc_ctx.state.playing:
         st.warning("Webcam stream is not running. Please start the stream.")
         # Reset analysis active state if stream stops
         if st.session_state.analysis_active:
             st.session_state.analysis_active = False


with col2:
    st.subheader("Estimated Metrics")
    metrics_placeholder = st.empty()

    st.subheader("Data Log")
    data_placeholder = st.empty()

    # --- Display Updates Outside the Callback ---
    # This part runs in the main Streamlit thread and updates the UI
    # based on the latest data stored in session_state (protected by lock)

    current_metrics = None
    current_data = pd.DataFrame() # Default empty df

    with lock: # Safely access shared state
         current_metrics = st.session_state.latest_metrics
         # Create a copy to avoid modifying shared state during iteration/display
         current_data = st.session_state.metrics_data.copy()


    # Update metrics visualization
    if current_metrics:
        fig = update_metrics_visualization(current_metrics)
        metrics_placeholder.pyplot(fig)
        plt.close(fig) # Close the figure
    else:
         # Display placeholder text if no metrics yet
         metrics_placeholder.info("Metrics will appear here once analysis starts and a face is detected.")


    # Update data table
    if not current_data.empty:
        data_placeholder.dataframe(current_data.round(2), use_container_width=True, height=250)
    else:
         data_placeholder.info("Logged data will appear here.")


# --- Cleanup (Optional but good practice) ---
# MediaPipe resources might be released automatically, but explicit close can be added if needed
# This is harder to guarantee with webrtc's background thread model compared to simple loop.
# Consider placing face_mesh.close() in appropriate cleanup logic if resource issues arise.
