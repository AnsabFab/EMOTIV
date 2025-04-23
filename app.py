import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import threading

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
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False
if 'face_landmarks' not in st.session_state:
    st.session_state.face_landmarks = None

# Title and description
st.title("Basic Facial Analysis")
st.markdown("""
This application analyzes facial landmarks to estimate psychological metrics:
- Emotional dimensions (valence, arousal, dominance)
- Personality traits and cognitive measures
""")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    # Camera feed
    st.subheader("Camera Feed")
    camera_placeholder = st.empty()
    
    # Camera device selection
    camera_devices = [0, 1, 2]  # Most systems will have camera 0
    selected_camera = st.selectbox("Select Camera", options=camera_devices, index=0)
    
    # Analysis frequency
    analysis_freq = st.slider("Analysis Frequency (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    
    # Start/Stop button
    if st.button("Start/Stop Analysis"):
        st.session_state.is_analyzing = not st.session_state.is_analyzing

with col2:
    # Metrics display
    st.subheader("Estimated Metrics")
    metrics_placeholder = st.empty()
    
    # Data table
    st.subheader("Data Log")
    data_placeholder = st.empty()

# Function to extract facial landmarks
def extract_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

# Function to calculate eye aspect ratio (EAR)
def calculate_ear(landmarks):
    if not landmarks:
        return 0.0
    
    # MediaPipe face mesh indices for the eyes
    LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Simplified for basic EAR calculation
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Simplified for basic EAR calculation
    
    # Get coordinates
    def get_landmark_coords(landmark_indices):
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in landmark_indices])
    
    left_eye_points = get_landmark_coords(LEFT_EYE)
    right_eye_points = get_landmark_coords(RIGHT_EYE)
    
    # Calculate EAR
    def eye_aspect_ratio(eye_points):
        # Compute vertical distances (height)
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute horizontal distance (width)
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Return the aspect ratio
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
    
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    
    # Average the EAR for both eyes
    return (left_ear + right_ear) / 2.0

# Function to calculate mouth aspect ratio (MAR)
def calculate_mar(landmarks):
    if not landmarks:
        return 0.0
    
    # MediaPipe face mesh indices for the mouth
    MOUTH_OUTLINE = [61, 291, 39, 181, 0, 17, 269, 405]
    
    # Get mouth landmark coordinates
    mouth_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in MOUTH_OUTLINE])
    
    # Calculate height and width
    height = np.mean([
        np.linalg.norm(mouth_points[1] - mouth_points[5]),
        np.linalg.norm(mouth_points[2] - mouth_points[6]),
        np.linalg.norm(mouth_points[3] - mouth_points[7])
    ])
    
    width = np.linalg.norm(mouth_points[0] - mouth_points[4])
    
    # Return aspect ratio
    return height / width if width > 0 else 0.0

# Function to calculate eyebrow position
def calculate_eyebrow_position(landmarks):
    if not landmarks:
        return 0.0
    
    # MediaPipe indices for eyebrows and eyes
    LEFT_EYEBROW = 107  # Center of left eyebrow
    RIGHT_EYEBROW = 336  # Center of right eyebrow
    LEFT_EYE = 159  # Center of left eye
    RIGHT_EYE = 386  # Center of right eye
    
    # Get y-coordinates
    left_eyebrow_y = landmarks.landmark[LEFT_EYEBROW].y
    right_eyebrow_y = landmarks.landmark[RIGHT_EYEBROW].y
    left_eye_y = landmarks.landmark[LEFT_EYE].y
    right_eye_y = landmarks.landmark[RIGHT_EYE].y
    
    # Calculate average distance (normalized)
    # Lower values mean eyebrows are raised (surprise/fear)
    # Higher values mean eyebrows are lowered (anger/concentration)
    left_distance = left_eye_y - left_eyebrow_y
    right_distance = right_eye_y - right_eyebrow_y
    
    # Average and normalize approximately (this will need calibration)
    avg_distance = (left_distance + right_distance) / 2.0
    
    # Typical range might be around 0.02-0.08, so we normalize
    normalized = (avg_distance - 0.02) / 0.06
    return max(0.0, min(1.0, normalized))

# Function to estimate head pose
def estimate_head_pose(landmarks):
    if not landmarks:
        return 0.0, 0.0
    
    # Get nose and eyes points
    NOSE_TIP = 4
    LEFT_EYE = 159
    RIGHT_EYE = 386
    
    nose = np.array([landmarks.landmark[NOSE_TIP].x, landmarks.landmark[NOSE_TIP].y, landmarks.landmark[NOSE_TIP].z])
    left_eye = np.array([landmarks.landmark[LEFT_EYE].x, landmarks.landmark[LEFT_EYE].y, landmarks.landmark[LEFT_EYE].z])
    right_eye = np.array([landmarks.landmark[RIGHT_EYE].x, landmarks.landmark[RIGHT_EYE].y, landmarks.landmark[RIGHT_EYE].z])
    
    # Calculate a simple vertical tilt (looking up/down)
    eye_level = (left_eye[1] + right_eye[1]) / 2.0
    vertical_tilt = nose[1] - eye_level
    
    # Calculate horizontal tilt (looking left/right)
    horizontal_mid = (left_eye[0] + right_eye[0]) / 2.0
    horizontal_tilt = nose[0] - horizontal_mid
    
    # Normalize (approximate)
    vertical_tilt = max(-1.0, min(1.0, vertical_tilt * 10))
    horizontal_tilt = max(-1.0, min(1.0, horizontal_tilt * 10))
    
    return vertical_tilt, horizontal_tilt

# Function to calculate all metrics based on facial landmarks
def calculate_metrics(landmarks):
    if not landmarks:
        # Return default values if no face is detected
        return {metric: 0.5 for metric in metrics}
    
    # Extract basic features
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    eyebrow_position = calculate_eyebrow_position(landmarks)
    vertical_tilt, horizontal_tilt = estimate_head_pose(landmarks)
    
    # Normalize and calculate simple cognitive load estimate
    # EAR decreases with cognitive load
    cognitive_load = max(0, min(1, 1.0 - ear * 2.5))  # Inverse relationship with EAR
    
    # Calculate approximate valence based on mouth shape and eyebrow position
    # Higher MAR and lower eyebrow = happier
    valence = max(0, min(1, mar * 2.0 * (1.0 - eyebrow_position)))
    
    # Calculate arousal based on overall facial activity
    arousal = max(0, min(1, (mar + (1.0 - ear) + eyebrow_position) / 3.0))
    
    # Calculate dominance based on head pose (higher when looking slightly down)
    dominance = max(0, min(1, 0.5 + vertical_tilt))
    
    # Calculate neuroticism (higher with higher cognitive load and lower valence)
    neuroticism = max(0, min(1, (cognitive_load * 0.6) + ((1.0 - valence) * 0.4)))
    
    # Emotional stability (inverse of neuroticism)
    emotional_stability = 1.0 - neuroticism
    
    # Extraversion (related to expressiveness, higher arousal and valence)
    extraversion = max(0, min(1, (arousal * 0.5) + (valence * 0.5)))
    
    # Openness (more expressive face)
    openness = max(0, min(1, 0.5 + ((mar - 0.5) * 0.5)))
    
    # Agreeableness (higher with positive valence)
    agreeableness = max(0, min(1, (valence * 0.7) + ((1.0 - arousal) * 0.3)))
    
    # Conscientiousness (steady patterns, moderate arousal)
    conscientiousness = max(0, min(1, (1.0 - abs(arousal - 0.5)) * 0.7 + (emotional_stability * 0.3)))
    
    # Stress index (combination of cognitive load, eyebrow position, low valence)
    stress_index = max(0, min(1, (cognitive_load * 0.5) + (eyebrow_position * 0.3) + ((1.0 - valence) * 0.2)))
    
    # Engagement level (combination of arousal and head position)
    engagement_level = max(0, min(1, (arousal * 0.7) + ((1.0 - abs(horizontal_tilt)) * 0.3)))
    
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

# Function to update metrics visualization
def update_metrics_visualization(metrics_data):
    # Create a figure with subplots for each metric
    fig, axs = plt.subplots(4, 3, figsize=(10, 8))
    axs = axs.flatten()
    
    # Custom colormap for the gauge charts
    colors = [(0.1, 0.1, 0.9), (0.9, 0.9, 0.1), (0.9, 0.1, 0.1)]  # Blue to Yellow to Red
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Create a gauge chart for each metric
    for i, (key, value) in enumerate(metrics_data.items()):
        if key == 'timestamp':
            continue
        
        # Create gauge chart
        axs[i].set_title(key.replace('_', ' ').title())
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(0, 0.5)
        axs[i].set_aspect('equal')
        axs[i].axis('off')
        
        # Draw the meter
        theta = np.linspace(0, np.pi, 100)
        r = 0.4  # radius
        
        # Background arc
        x_bg = 0.5 + r * np.cos(theta)
        y_bg = 0.1 + r * np.sin(theta)
        axs[i].plot(x_bg, y_bg, 'k-', linewidth=3, alpha=0.3)
        
        # Value arc
        value_theta = np.linspace(0, np.pi * value, 100)
        x_val = 0.5 + r * np.cos(value_theta)
        y_val = 0.1 + r * np.sin(value_theta)
        
        points = np.array([x_val, y_val]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.linspace(0, 1, len(segments)))
        lc.set_linewidth(5)
        axs[i].add_collection(lc)
        
        # Add the value text
        axs[i].text(0.5, 0.25, f"{value:.2f}", ha='center', va='center', fontsize=12, 
                  fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    return fig

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
        
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks if analyzing
        current_time = time.time()
        if st.session_state.is_analyzing and (current_time - last_analysis_time >= analysis_freq):
            # Extract landmarks
            landmarks = extract_face_landmarks(frame)
            st.session_state.face_landmarks = landmarks
            
            # Calculate metrics
            if landmarks:
                calculated_metrics = calculate_metrics(landmarks)
                
                # Add timestamp
                elapsed_time = current_time - st.session_state.start_time
                current_metrics = {
                    'timestamp': elapsed_time,
                    **calculated_metrics
                }
                
                # Update dataframe
                new_row = pd.DataFrame([current_metrics])
                st.session_state.metrics_data = pd.concat([st.session_state.metrics_data, new_row], ignore_index=True)
                
                # Update metrics visualization
                fig = update_metrics_visualization(calculated_metrics)
                metrics_placeholder.pyplot(fig)
                
                # Update data table
                data_placeholder.dataframe(st.session_state.metrics_data.round(2), use_container_width=True)
            
            last_analysis_time = current_time
        
        # Draw face mesh if available
        if st.session_state.face_landmarks:
            # Draw landmarks on frame
            annotated_frame = frame_rgb.copy()
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=st.session_state.face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw eyes, eyebrows, and lips contours for better visibility
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=st.session_state.face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            camera_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
        else:
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Sleep to control frame rate
        time.sleep(0.05)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
finally:
    # Clean up resources
    if 'cap' in locals() and cap is not None:
        cap.release()
