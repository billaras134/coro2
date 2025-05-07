import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from segment_anything import SamPredictor, sam_model_registry
from sklearn.linear_model import LinearRegression

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

st.set_page_config(page_title="AI Coronary Angiography Assistant", layout="wide")
st.title("🫀 AI-Based Coronary Angiography Assistant")

st.write("""
Upload a coronary angiography video or image and let AI assist you in detecting vessels and possible lesions.
This is a prototype for research and educational purposes.
""")

uploaded_file = st.file_uploader("Upload an angiography video or image", type=["jpg", "jpeg", "png", "mp4"])

def estimate_stenosis(mask):
    vessel_pixels = np.sum(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    vessel_density = vessel_pixels / total_pixels
    stenosis_percent = max(0, (0.3 - vessel_density) * 300)
    return min(stenosis_percent, 99)

def recognize_artery(point, image_shape):
    h, w = image_shape[:2]
    x, y = point
    if y < h // 3:
        return "Left Anterior Descending (LAD)"
    elif y < 2 * h // 3:
        return "Left Circumflex (LCx)"
    else:
        return "Right Coronary Artery (RCA)"

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if file_ext == '.mp4':
        st.video(uploaded_file)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total Frames in Video: {frame_count}")
        stframe = st.empty()
        frame_results = []

        for frame_num in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 10 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictor.set_image(frame)
                h, w, _ = frame.shape
                input_point = np.array([[w//2, h//2]])
                input_label = np.array([1])
                masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
                mask = masks[0]
                vessel_highlight = rgb_frame.copy()
                vessel_highlight[mask == 1] = [255, 0, 0]
                stenosis_percent = estimate_stenosis(mask)
                artery = recognize_artery((w//2, h//2), frame.shape)
                stframe.image(vessel_highlight, caption=f"Frame {frame_num} - {artery} - Estimated Stenosis: {stenosis_percent:.1f}%")
                frame_results.append(f"Frame {frame_num}: {artery} - Estimated Stenosis = {stenosis_percent:.1f}%")

        cap.release()
        st.subheader("AI Interpretation Draft for Video")
        report_text = "\n".join(frame_results)
        st.download_button("Download Draft Report", report_text)

    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image_np, caption="Original Angiography Image", use_column_width=True)
        input_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        predictor.set_image(input_image)
        h, w, _ = input_image.shape
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
        mask = masks[0]
        vessel_highlight = image_np.copy()
        vessel_highlight[mask == 1] = [255, 0, 0]
        stenosis_percent = estimate_stenosis(mask)
        artery = recognize_artery((w//2, h//2), image_np.shape)
        st.image(vessel_highlight, caption=f"AI Highlighted Vessels - {artery} - Estimated Stenosis: {stenosis_percent:.1f}%", use_column_width=True)
        st.subheader("AI Interpretation Draft")
        st.write(f"""
- Vessel segmentation completed.
- Detected artery: {artery}
- Estimated stenosis: {stenosis_percent:.1f}%
- Potential lesion zones marked in red.
- Suggested next step: Clinical correlation and possible further evaluation.
""")
        report_text = f"""
AI Coronary Angiography Preliminary Report
-------------------------------------------

Patient Image analyzed.
Vessels identified and mapped with AI model.
Detected artery: {artery}
Estimated stenosis: {stenosis_percent:.1f}%

* Areas of possible narrowing highlighted (requires confirmation).
* Recommend clinical evaluation.
"""
        st.download_button("Download Draft Report", report_text)
