import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.face_detection import FaceDetection

# Enhanced facial landmark indices for different facial features
face_points = {
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463],
    "FACE": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152],
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    "EYEBROW_LEFT": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
    "EYEBROW_RIGHT": [285, 336, 296, 334, 293, 300, 276, 283, 295, 285],
}

# Color presets (RGB format)
LIP_COLORS = {
    "Red": [0, 0, 255],
    "Pink": [147, 20, 255],
    "brown": [10, 30, 80],
    "voilet": [121, 13, 142],
    "Coral": [255, 131, 93]
}

BLUSH_COLORS = {
    "Red": [0, 0, 255],
    "Pink": [147, 20, 255],
    "brown": [127, 15, 15],
    "voilet": [121, 13, 142],
    "Coral": [255, 131, 93]
}

EYELINER_COLORS = {
    "Black": [14, 4, 4],
    "Brown": [10, 30, 80],
    "Navy": [0, 0, 128],
    "Purple": [128, 0, 128],
}

EYESHADOW_COLORS = {
    "Black": [14, 4, 4],
    "Brown": [165, 42, 42],
    "Navy": [0, 0, 128],
    "Purple": [128, 0, 128],
    "Pink": [255, 20, 147]
}

FOUNDATION_INTENSITIES = {
    "Light": 1.2,
    "Medium": 1.5,
    "Full": 1.75
}

# Landmark functions
def detect_landmarks(src, is_stream=False):
    """Given an image src retrieves the facial landmarks associated with it"""
    with FaceMesh(static_image_mode=not is_stream, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

def normalize_landmarks(landmarks, height, width, indices=None):
    """Normalizes landmarks to image dimensions"""
    if not landmarks:
        return None
    normalized_landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])
    if indices:
        return normalized_landmarks[indices]
    return normalized_landmarks

def plot_landmarks(src, landmarks, show=False):
    """Plots landmarks on the image"""
    dst = src.copy()
    for x, y in landmarks:
        cv2.circle(dst, (x, y), 2, (0, 0, 0), cv2.FILLED)
    return dst

# Makeup functions
def lip_mask(src, points, color):
    """Creates a lip color mask"""
    mask = np.zeros_like(src)
    upper_points = points[:len(face_points["LIP_UPPER"])]
    lower_points = points[len(face_points["LIP_UPPER"]):]
    
    # Convert RGB to BGR for OpenCV
    opencv_color = [color[2], color[1], color[0]]
    
    # Create separate masks for upper and lower lip for better control
    mask = cv2.fillPoly(mask, [upper_points], opencv_color)
    mask = cv2.fillPoly(mask, [lower_points], opencv_color)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask

def blush_mask(src, left_points, right_points, color, radius=50):
    """Creates a blush mask for both cheeks"""
    mask = np.zeros_like(src)
    
    # Convert RGB to BGR for OpenCV
    opencv_color = [color[2], color[1], color[0]]
    
    # Apply blush to left cheek
    for point in left_points:
        mask = cv2.circle(mask, point, radius, opencv_color, cv2.FILLED)
        x, y = point[0] - radius, point[1] - radius
        roi = mask[max(0, y):min(src.shape[0], y + 2 * radius), max(0, x):min(src.shape[1], x + 2 * radius)]
        if roi.size > 0:  # Ensure ROI is not empty
            mask[max(0, y):min(src.shape[0], y + 2 * radius), max(0, x):min(src.shape[1], x + 2 * radius)] = vignette(roi, 10)
    
    # Apply blush to right cheek
    for point in right_points:
        mask = cv2.circle(mask, point, radius, opencv_color, cv2.FILLED)
        x, y = point[0] - radius, point[1] - radius
        roi = mask[max(0, y):min(src.shape[0], y + 2 * radius), max(0, x):min(src.shape[1], x + 2 * radius)]
        if roi.size > 0:  # Ensure ROI is not empty
            mask[max(0, y):min(src.shape[0], y + 2 * radius), max(0, x):min(src.shape[1], x + 2 * radius)] = vignette(roi, 10)
    
    return mask

def eyeliner_mask(src, left_points, right_points, color, thickness=2):
    """Creates an eyeliner mask"""
    mask = np.zeros_like(src)
    
    # Convert RGB to BGR for OpenCV
    opencv_color = [color[2], color[1], color[0]]
    
    # Draw eyeliner for left eye
    for i in range(len(left_points) - 1):
        cv2.line(mask, left_points[i], left_points[i + 1], opencv_color, thickness)
    
    # Draw eyeliner for right eye
    for i in range(len(right_points) - 1):
        cv2.line(mask, right_points[i], right_points[i + 1], opencv_color, thickness)
    
    # Add slight blur for more natural look
    mask = cv2.GaussianBlur(mask, (3, 3), 1)
    
    return mask

def eyeshadow_mask(src, left_points, right_points, color):
    """Creates an eyeshadow mask"""
    mask = np.zeros_like(src)
    
    # Convert RGB to BGR for OpenCV
    opencv_color = [color[2], color[1], color[0]]
    
    # Apply eyeshadow to left eye
    mask = cv2.fillPoly(mask, [left_points], opencv_color)
    
    # Apply eyeshadow to right eye
    mask = cv2.fillPoly(mask, [right_points], opencv_color)
    
    # Blur for more natural look
    mask = cv2.GaussianBlur(mask, (15, 15), 10)
    
    return mask

def mask_skin(src):
    """Creates a skin mask for foundation"""
    lower = np.array([150, 60, 98], dtype='uint8')
    upper = np.array([150, 60, 98], dtype='uint8')
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    skin_mask = cv2.inRange(dst, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # Ensure skin_mask is just a 2D array (height, width) not a 3D array
    if skin_mask.ndim == 3:
        skin_mask = skin_mask[:,:,0]
    
    return skin_mask

def gamma_correction(src, gamma, coefficient=1):
    """Performs gamma correction"""
    dst = src.copy().astype(np.float64) / 255.0
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

def vignette(src, sigma):
    """Creates a vignette effect"""
    height, width = src.shape[:2]
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)
    
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    if src.ndim == 3:
        blurred = cv2.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    else:
        blurred = cv2.convertScaleAbs(src.copy() * mask)
    return blurred

def face_bbox(src, offset_x=0, offset_y=0):
    """Gets face bounding box"""
    height, width = src.shape[:2]
    with FaceDetection(model_selection=0) as detector:
        results = detector.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
    results = results.detections[0].location_data
    x_min, y_min = results.relative_bounding_box.xmin, results.relative_bounding_box.ymin
    box_height, box_width = results.relative_bounding_box.height, results.relative_bounding_box.width
    x_min = int(width * x_min) - offset_x
    y_min = int(height * y_min) - offset_y
    box_height, box_width = int(height * box_height) + offset_y, int(width * box_width) + offset_x
    return (x_min, y_min), (box_height, box_width)

def apply_makeup(src, is_stream, feature, color, intensity=None, thickness=2, show_landmarks=False):
    """Applies makeup to the image"""
    ret_landmarks = detect_landmarks(src, is_stream)
    if ret_landmarks is None:
        st.error("No face detected in the image. Please try again.")
        return src
        
    height, width = src.shape[:2]
    normalized_landmarks = normalize_landmarks(ret_landmarks, height, width)
    
    if feature == 'lips':
        upper_lip_points = [normalized_landmarks[i] for i in face_points["LIP_UPPER"]]
        lower_lip_points = [normalized_landmarks[i] for i in face_points["LIP_LOWER"]]
        lip_points = np.array(upper_lip_points + lower_lip_points)
        mask = lip_mask(src, lip_points, color)
        output = cv2.addWeighted(src, 1.0, mask, 0.5, 0.0)
        
    elif feature == 'blush':
        left_cheek_points = [normalized_landmarks[i] for i in face_points["BLUSH_LEFT"]]
        right_cheek_points = [normalized_landmarks[i] for i in face_points["BLUSH_RIGHT"]]
        mask = blush_mask(src, left_cheek_points, right_cheek_points, color, 60)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)
        
    elif feature == 'foundation':
        skin_mask = mask_skin(src)
        # Creating a 3-channel mask to match the 3-channel images
        skin_mask_3d = np.repeat(np.expand_dims(skin_mask, axis=2), 3, axis=2)
        # Apply foundation where skin is detected
        corrected_img = gamma_correction(src, intensity)
        output = np.where(skin_mask_3d > 0, corrected_img, src)
        
    elif feature == 'eyeliner':
        left_eyeliner_points = [normalized_landmarks[i] for i in face_points["EYELINER_LEFT"]]
        right_eyeliner_points = [normalized_landmarks[i] for i in face_points["EYELINER_RIGHT"]]
        mask = eyeliner_mask(src, left_eyeliner_points, right_eyeliner_points, color, thickness)
        output = cv2.addWeighted(src, 1.0, mask, 0.8, 0.0)
        
    elif feature == 'eyeshadow':
        left_eyeshadow_points = [normalized_landmarks[i] for i in face_points["EYESHADOW_LEFT"]]
        right_eyeshadow_points = [normalized_landmarks[i] for i in face_points["EYESHADOW_RIGHT"]]
        mask = eyeshadow_mask(src, np.array(left_eyeshadow_points), np.array(right_eyeshadow_points), color)
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)
        
    else:
        output = src
        
    if show_landmarks:
        # Draw facial landmarks for debugging
        for i, point in enumerate(normalized_landmarks):
            x, y = point
            cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(output, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
    return output

# Streamlit app
def main():
    st.set_page_config(page_title="Virtual Makeup App", layout="wide")

    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #FF1493;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #FF69B4;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .feature-box {
        background-color: #FFF0F5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>✨ Virtual Makeup Studio ✨</h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'makeup_applied' not in st.session_state:
        st.session_state.makeup_applied = False
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'capture_clicked' not in st.session_state:
        st.session_state.capture_clicked = False
    if 'stop_clicked' not in st.session_state:
        st.session_state.stop_clicked = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # Sidebar for selection
    st.sidebar.markdown("<h2 class='sub-header'>Makeup Options</h2>", unsafe_allow_html=True)
    
    # Image source selection
    image_source = st.sidebar.radio("Choose input method:", ["Upload Image", "Selfie Mode"])
    
    # Image container placeholder
    image_placeholder = st.empty()
    camera_placeholder = st.empty()
    
    # Image upload/capture - Fixed collision issues
    if image_source == "Upload Image":
        # Stop camera if running when switching to upload
        if st.session_state.camera_running:
            st.session_state.camera_running = False
            st.session_state.capture_clicked = False
            st.session_state.stop_clicked = False
            
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
        
        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            image = Image.open(uploaded_file)
            image = np.array(image)
            # Convert RGBA to RGB if needed
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            st.session_state.current_image = image
            st.session_state.original_image = image.copy()
            st.session_state.makeup_applied = False
            
    else:  # Selfie Mode
        # Clear upload state when switching to camera
        st.session_state.uploaded_file = None
        
        # Define button callbacks
        def start_camera():
            st.session_state.camera_running = True
            
        def capture_image():
            st.session_state.capture_clicked = True
            
        def stop_camera():
            st.session_state.stop_clicked = True
            st.session_state.camera_running = False
        
        # Camera control buttons
        if not st.session_state.camera_running:
            st.sidebar.button("Start Camera", on_click=start_camera, key="start_camera")
        else:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.button("Capture", on_click=capture_image, key="capture")
            with col2:
                st.button("Stop", on_click=stop_camera, key="stop")
        
        # Camera logic
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.camera_running = False
            else:
                # Single frame capture outside of loop to avoid the infinite loop issue
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)  # Flip horizontally for selfie view
                    camera_placeholder.image(frame, channels="RGB", caption="Camera Feed")
                    
                    # Handle capture click
                    if st.session_state.capture_clicked:
                        st.session_state.current_image = frame.copy()
                        st.session_state.original_image = frame.copy()
                        st.session_state.makeup_applied = False
                        st.session_state.capture_clicked = False
                        st.session_state.camera_running = False
                        st.success("Image captured!")
                else:
                    st.error("Failed to get frame from webcam")
                    st.session_state.camera_running = False
                
                # Handle stop click
                if st.session_state.stop_clicked:
                    st.session_state.stop_clicked = False
                
                cap.release()

    # Display the current image
    if st.session_state.current_image is not None:
        if not st.session_state.makeup_applied:
            image_placeholder.image(st.session_state.current_image, channels="RGB", caption="Original Image")
        else:
            image_placeholder.image(st.session_state.current_image, channels="RGB", caption="Image with Makeup Applied")
        
        # Makeup application options
        st.sidebar.markdown("<h2 class='sub-header'>Apply Makeup</h2>", unsafe_allow_html=True)
        
        # Create tabs for different makeup categories
        tabs = st.sidebar.tabs(["Face", "Eyes", "Lips"])
        
        with tabs[0]:  # Face tab
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            apply_foundation = st.checkbox("Apply Foundation", key="foundation")
            if apply_foundation:
                foundation_intensity = st.selectbox("Foundation Coverage", list(FOUNDATION_INTENSITIES.keys()), key="foundation_intensity")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            apply_blush = st.checkbox("Apply Blush", key="blush")
            if apply_blush:
                blush_color = st.selectbox("Blush Color", list(BLUSH_COLORS.keys()), key="blush_color")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tabs[1]:  # Eyes tab
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            apply_eyeliner = st.checkbox("Apply Eyeliner", key="eyeliner")
            if apply_eyeliner:
                eyeliner_color = st.selectbox("Eyeliner Color", list(EYELINER_COLORS.keys()), key="eyeliner_color")
                eyeliner_thickness = st.slider("Eyeliner Thickness", 1, 5, 2, key="eyeliner_thickness")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            apply_eyeshadow = st.checkbox("Apply Eyeshadow", key="eyeshadow")
            if apply_eyeshadow:
                eyeshadow_color = st.selectbox("Eyeshadow Color", list(EYESHADOW_COLORS.keys()), key="eyeshadow_color")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tabs[2]:  # Lips tab
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            apply_lips = st.checkbox("Apply Lipstick", key="lips")
            if apply_lips:
                lip_color = st.selectbox("Lipstick Color", list(LIP_COLORS.keys()), key="lip_color")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Debug option
        show_landmarks = st.sidebar.checkbox("Show Face Landmarks (Debug)", key="show_landmarks")
        
        # Apply makeup button
        if st.sidebar.button("Apply Makeup", key="apply_makeup"):
            # Start with the original image
            result_image = st.session_state.original_image.copy()
            
            # Apply makeup in sequence: foundation first, then other features
            if apply_foundation:
                intensity = FOUNDATION_INTENSITIES[foundation_intensity]
                result_image = apply_makeup(result_image, False, 'foundation', None, intensity, show_landmarks=False)
                
            if apply_blush:
                color = BLUSH_COLORS[blush_color]
                result_image = apply_makeup(result_image, False, 'blush', color, show_landmarks=False)
            
            if apply_eyeshadow:
                color = EYESHADOW_COLORS[eyeshadow_color]
                result_image = apply_makeup(result_image, False, 'eyeshadow', color, show_landmarks=False)
                
            if apply_eyeliner:
                color = EYELINER_COLORS[eyeliner_color]
                result_image = apply_makeup(result_image, False, 'eyeliner', color, thickness=eyeliner_thickness, show_landmarks=False)
                
            if apply_lips:
                color = LIP_COLORS[lip_color]
                result_image = apply_makeup(result_image, False, 'lips', color, show_landmarks=False)
            
            # Apply landmarks last if requested
            if show_landmarks:
                landmarks = detect_landmarks(result_image, False)
                if landmarks:
                    normalized = normalize_landmarks(landmarks, result_image.shape[0], result_image.shape[1])
                    for i, (x, y) in enumerate(normalized):
                        cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)
                        # Only show some landmark indices to avoid clutter
                        if i % 10 == 0:
                            cv2.putText(result_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            st.session_state.current_image = result_image
            st.session_state.makeup_applied = True
            
            # Update the displayed image
            image_placeholder.image(result_image, channels="RGB", caption="Image with Makeup Applied")
        
        # Reset button
        if st.sidebar.button("Reset Image", key="reset_image"):
            st.session_state.current_image = st.session_state.original_image.copy()
            st.session_state.makeup_applied = False
            image_placeholder.image(st.session_state.current_image, channels="RGB", caption="Original Image")
            
        # Download button for the result
        if st.session_state.makeup_applied:
            # Convert the image to bytes
            result_pil = Image.fromarray(st.session_state.current_image)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.sidebar.download_button(
                label="Download Result",
                data=byte_im,
                file_name="makeup_result.png",
                mime="image/png",
                key="download"
            )

    # Instructions and information
    with st.expander("How to Use"):
        st.markdown("""
        ### Instructions:
        
        1. **Choose Input Method**:
           - Upload an image from your device
           - Use your webcam in selfie mode
           
        2. **Select Makeup Options**:
           - Navigate through tabs to select face, eye, and lip makeup
           - Check the boxes for the features you want to apply
           - Choose colors and intensities for each feature
           
        3. **Apply and Download**:
           - Click "Apply Makeup" to see the result
           - Use "Reset Image" to start over
           - Download your final image when satisfied
           
        ### Tips:
        - For best results, use images with clear, front-facing portraits
        - Make sure your face is well-lit when using selfie mode
        - Try different combinations of makeup for different looks!
        - If you want to see the facial landmarks (for technical purposes), use the debug option
        """)

if __name__ == "__main__":
    main()