import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection # Added specific import
# import threading # Threading is not used in this version, can be removed if not needed elsewhere

# Set page config
st.set_page_config(
    page_title="Basic Facial Analysis",
    page_icon="😀",
    layout="wide"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Initialize Face Mesh - Consider moving inside if settings change often, but usually fine here
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

# Initialize session state variables if they don't exist
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = pd.DataFrame(columns=['timestamp'] + metrics)
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False
if 'face_landmarks' not in st.session_state:
    st.session_state.face_landmarks = None
if 'camera_error' not in st.session_state:
    st.session_state.camera_error = False

# Title and description
st.title("Basic Facial Analysis")
st.markdown("""
This application attempts to analyze facial landmarks from a webcam feed to estimate various psychological metrics.
*Note: These estimations are based on simplified correlations and are for demonstrative purposes only. They are not scientifically validated psychological assessments.*
""")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    # Camera feed
    st.subheader("Camera Feed")
    camera_placeholder = st.empty() # Placeholder for the video frame

    # Camera device selection
    # Common indices, might need adjustment based on the system
    camera_devices = [0, 1, 2, 3]
    selected_camera = st.selectbox("Select Camera", options=camera_devices, index=0, key="camera_select")

    # Analysis frequency
    analysis_freq = st.slider("Analysis Frequency (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    # Start/Stop button
    if st.button("Start/Stop Analysis"):
        st.session_state.is_analyzing = not st.session_state.is_analyzing
        # Reset landmarks if stopping analysis
        if not st.session_state.is_analyzing:
            st.session_state.face_landmarks = None

    # Display current status
    status_text = "Analyzing" if st.session_state.is_analyzing else "Stopped"
    st.info(f"Status: {status_text}")


with col2:
    # Metrics display
    st.subheader("Estimated Metrics")
    metrics_placeholder = st.empty() # Placeholder for the gauge charts

    # Data table
    st.subheader("Data Log")
    data_placeholder = st.empty() # Placeholder for the dataframe

# --- Analysis Functions (Keep these exactly as you provided) ---

# Function to extract facial landmarks
def extract_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

# Function to calculate eye aspect ratio (EAR)
def calculate_ear(landmarks):
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

# Function to calculate mouth aspect ratio (MAR)
def calculate_mar(landmarks):
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

# Function to calculate eyebrow position
def calculate_eyebrow_position(landmarks):
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

# Function to estimate head pose
def estimate_head_pose(landmarks):
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

# Function to calculate all metrics based on facial landmarks
def calculate_metrics(landmarks):
    if not landmarks:
        return {metric: 0.5 for metric in metrics} # Return default values
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    eyebrow_position = calculate_eyebrow_position(landmarks)
    vertical_tilt, horizontal_tilt = estimate_head_pose(landmarks)

    # --- Simplified Metric Estimations (Keep as is) ---
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

# Function to update metrics visualization (gauge charts)
def update_metrics_visualization(metrics_values):
    if not metrics_values: # Handle case where metrics haven't been calculated yet
         # Create a blank figure or return None
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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        r = 0.4 # radius
        theta = np.linspace(np.pi, 0, 100) # Flipped for gauge direction
        x_bg = 0.5 + r * np.cos(theta)
        y_bg = 0.1 + r * np.sin(theta)
        ax.plot(x_bg, y_bg, 'k-', linewidth=3, alpha=0.2) # Background arc

        # Value arc needs careful calculation to map value [0,1] to angle [pi, 0]
        value_angle = np.pi * (1 - value)
        value_theta = np.linspace(np.pi, value_angle, 100)
        x_val = 0.5 + r * np.cos(value_theta)
        y_val = 0.1 + r * np.sin(value_theta)

        # Create line segments for coloring
        points = np.array([x_val, y_val]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Map segment colors based on the *value* they represent (not just position)
        segment_values = np.linspace(0, value, len(segments))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(segment_values)
        lc.set_linewidth(5)
        ax.add_collection(lc)

        ax.text(0.5, 0.15, f"{value:.2f}", ha='center', va='center', fontsize=11,
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        metric_idx += 1

    # Hide unused subplots
    for i in range(metric_idx, len(axs)):
        axs[i].axis('off')

    plt.tight_layout(pad=0.5) # Add padding
    return fig

# --- Main Execution Block ---

cap = None # Initialize cap outside try
try:
    # Attempt to initialize the camera selected by the user
    cap = cv2.VideoCapture(selected_camera)
    if not cap.isOpened():
        st.error(f"Error: Could not open video device {selected_camera}. It might be in use or unavailable.")
        st.session_state.camera_error = True
    else:
        st.session_state.camera_error = False
        st.success(f"Camera {selected_camera} opened successfully.")

    last_analysis_time = time.time()
    latest_metrics = None # Store the latest calculated metrics

    # Only run the loop if the camera was opened successfully
    if not st.session_state.camera_error and cap is not None:
        while True:
            # Read frame from camera
            ret, frame = cap.read()

            if not ret:
                st.warning("Warning: Failed to capture frame from camera. Stream might have ended.")
                # Optionally break or try to reconnect depending on desired behavior
                time.sleep(0.5) # Pause before trying again or stopping
                # Re-check if camera is opened, might need re-initialization logic here if needed
                if not cap.isOpened():
                     st.error("Camera stream lost.")
                     break # Exit the loop if camera is truly gone
                continue # Try reading next frame


            # Convert frame BGR to RGB for display and mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = frame_rgb.copy() # Work on a copy for drawing

            # --- Perform Analysis Periodically ---
            current_time = time.time()
            if st.session_state.is_analyzing and (current_time - last_analysis_time >= analysis_freq):
                # Extract landmarks
                landmarks = extract_face_landmarks(frame) # Use original frame for analysis
                st.session_state.face_landmarks = landmarks # Store landmarks (or None)

                # Calculate metrics ONLY if landmarks were found
                if landmarks:
                    calculated_metrics = calculate_metrics(landmarks)
                    latest_metrics = calculated_metrics # Update latest metrics

                    # Add timestamp relative to start
                    elapsed_time = current_time - st.session_state.start_time
                    current_metrics_row = {
                        'timestamp': elapsed_time,
                        **calculated_metrics
                    }

                    # Update dataframe (use concat as before)
                    new_row = pd.DataFrame([current_metrics_row])
                    st.session_state.metrics_data = pd.concat([st.session_state.metrics_data, new_row], ignore_index=True)

                else:
                    # If no face detected during analysis, clear landmarks but keep last known metrics for display? Or reset?
                    # Option 1: Keep showing last known metrics - 'latest_metrics' remains unchanged
                    # Option 2: Show default/neutral metrics if no face - uncomment below
                    # latest_metrics = {metric: 0.5 for metric in metrics}
                    pass # Keep last known metrics for now

                last_analysis_time = current_time

            # --- Update Visualizations in Place ---

            # Draw face mesh ONLY if landmarks are available from the latest successful analysis
            if st.session_state.face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=st.session_state.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=st.session_state.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Display the (potentially annotated) frame
            camera_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

            # Update metrics display (gauge charts) - uses 'latest_metrics'
            # Only update if we have some metrics to show
            if latest_metrics:
                fig = update_metrics_visualization(latest_metrics)
                metrics_placeholder.pyplot(fig)
                plt.close(fig) # Close the figure to free memory


            # Update data table
            # Only display if there's data
            if not st.session_state.metrics_data.empty:
                 data_placeholder.dataframe(st.session_state.metrics_data.round(2), use_container_width=True, height=200) # Limit height


            # Small sleep to prevent extremely high CPU usage and allow Streamlit to process events
            time.sleep(0.01) # Shorter sleep for smoother video

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.error("Please ensure your webcam is connected and permissions are granted.")
    import traceback
    st.error(traceback.format_exc()) # Print full traceback for debugging

finally:
    # Clean up resources
    if cap is not None and cap.isOpened():
        cap.release()
        print("Camera released.") # For debugging in console
    if 'face_mesh' in locals():
         face_mesh.close() # Close mediapipe resources
