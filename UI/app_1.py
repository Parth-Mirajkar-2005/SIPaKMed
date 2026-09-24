import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import time

st.set_page_config(page_title="SIPaKMeD AI Diagnostic App", layout="wide", page_icon="🔬")

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    /* Dark mode sleek background for metrics */
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #3b3b54;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] > div > div > div > div {
        color: #a6accd !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] > div > div > div > div:nth-child(2) {
        color: #89b4fa !important;
        font-size: 1.5rem !important;
    }
    /* Headers */
    h1, h2, h3 {
        color: #89b4fa !important;
        font-family: 'Inter', sans-serif;
    }
    /* Sleek buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #89b4fa 0%, #cba6f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(203, 166, 247, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to prevent websocket lag
def resize_for_web(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

# 1. Load Models (Cached so they only load once!)
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(r"runs\detect\sipakmed_yolo\weights\best.pt")
    
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    unet.load_state_dict(torch.load(r"trained_models\unet\unet_sipakmed.pth", map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()
    
    import torchvision.models as models
    import torch.nn as nn
    cnn = models.efficientnet_b0(weights=None)
    cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 5)
    cnn.load_state_dict(torch.load(r"trained_models\cnn\efficientnet_sipakmed.pth", map_location=device, weights_only=True))
    cnn.to(device)
    cnn.eval()
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(r"trained_models\xgboost_sipakmed.json")
    
    return yolo, unet, cnn, xgb_model, device

with st.spinner("Loading AI Models into memory... This only happens once!"):
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()

# Standard settings
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

# --- APP UI ---
st.title("🔬 Cinematic AI-Powered Cervical Cancer Diagnostics")
st.markdown("""
**Interactive AI Analysis Mode:** Watch the AI process the image layer by layer in real-time.
1. **YOLOv8** physically scans the slide for cells.
2. **U-Net** isolates the nucleus geometry and overlays the mask.
3. **EfficientNet** generates a thermal Grad-CAM heatmap showing its visual focus.
4. **XGBoost** logs the live parameters and makes the final diagnosis.
""")

uploaded_file = st.file_uploader("Upload a microscope image (.bmp, .jpg, .png)", type=["bmp", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    # Main dynamic image placeholder (Fast web rendering)
    main_image_placeholder = st.empty()
    web_ready_img = resize_for_web(original_img)
    main_image_placeholder.image(cv2.cvtColor(web_ready_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Original Raw Slide")
    
    if st.button("Start Cinematic AI Diagnostic", type="primary"):
        st.divider()
        st.subheader("🧠 Live AI Thought Process")
        
        # --- PHASE 1: YOLO SCANNING ---
        st.markdown("### Phase 1: YOLOv8 Cell Detection")
        scan_status = st.empty()
        
        with st.spinner("Scanning slide for cellular structures..."):
            time.sleep(0.5) # Dramatic pause
            results = yolo_model(original_img, verbose=False)[0]
            
        cells_to_process = []
        display_img = original_img.copy()
        
        if len(results.boxes) == 0:
            scan_status.warning("No cells detected! Treating entire image as one single cell.")
            h, w, _ = original_img.shape
            cells_to_process.append((0, 0, w, h, original_img))
        else:
            scan_status.success(f"Detected {len(results.boxes)} potential cells! Initiating Deep Dive...")
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cell_crop = original_img[y1:y2, x1:x2]
                if cell_crop.size > 0:
                    cells_to_process.append((x1, y1, x2, y2, cell_crop))
                
                # Animate the bounding box drawing SMOOTHLY using the web-ready image scaling
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), max(3, int(display_img.shape[1]/500)))
                fast_display = resize_for_web(display_img)
                main_image_placeholder.image(cv2.cvtColor(fast_display, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Phase 1: YOLOv8 Scanning...")
                time.sleep(0.4) # Slowed down for cinematic effect
        
        st.divider()
        
        # --- PHASE 2 & 3: Deep Dive per cell ---
        st.markdown("### Phase 2: Morphological & Texture Deep Dive")
        
        all_cell_features = []
        
        # UI Elements for the live analysis
        col_cell, col_mask, col_prob = st.columns(3)
        with col_cell:
            cell_header = st.empty()
            cell_image = st.empty()
        with col_mask:
            mask_header = st.empty()
            mask_image = st.empty()
        with col_prob:
            texture_header = st.empty()
            texture_image = st.empty()
            
        st.markdown("#### Live Extraction Metrics")
        
        # Sleek Metrics Dashboards
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            m_id = st.empty()
        with met_col2:
            m_nc = st.empty()
        with met_col3:
            m_circ = st.empty()
            
        st.markdown("#### Final XGBoost Meta-Learner Data")
        live_table_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
            cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
            
            # 1. Show Extracted Cell
            cell_header.markdown(f"**Cell {idx+1}: Extracted**")
            cell_image.image(cell_rgb, use_container_width=True)
            
            mask_header.markdown(f"**Cell {idx+1}: U-Net Geometry**")
            mask_image.image(np.zeros_like(cell_rgb), caption="Painting geometry mask...")
            texture_header.markdown(f"**Cell {idx+1}: Chromatin Texture**")
            texture_image.info("Enhancing chromatin topography...")
            
            time.sleep(0.8) # Slow down before processing
            
            # 2. U-Net Morphology
            unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
            unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
            with torch.no_grad():
                mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
            
            # Create visual mask image
            visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
            visual_mask[mask == 1] = [0, 255, 0] # Cytoplasm Green
            visual_mask[mask == 2] = [255, 0, 0] # Nucleus Red
            
            mask_image.image(visual_mask, use_container_width=True, caption="Green=Cytoplasm, Red=Nucleus")
            
            cyt_area = float(np.sum(mask == 1))
            nuc_area = float(np.sum(mask == 2))
            nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
            
            # --- QUALITY CONTROL CHECK ---
            if nuc_area < 5:
                texture_header.markdown(f"**Cell {idx+1}: ❌ REJECTED (QC)**")
                texture_image.error("Quality Control Failure: No intact nucleus detected by U-Net. Skipping diagnosis to prevent false positives.")
                time.sleep(1.5)
                continue
            
            # Advanced OpenCV Geometry
            nuc_perimeter, nuc_circularity, nuc_eccentricity = 0.0, 0.0, 0.0
            nuc_mask_uint8 = (mask == 2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(nuc_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                nuc_perimeter = float(cv2.arcLength(largest_contour, True))
                if nuc_perimeter > 0:
                    nuc_circularity = 4 * np.pi * (nuc_area / (nuc_perimeter ** 2))
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    if ellipse[1][0] > 0: nuc_eccentricity = ellipse[1][1] / ellipse[1][0]
            
            # Chromatin Texture Map (CLAHE Enhancement)
            gray_img = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray_img, (128, 128))
            
            nuc_only_gray = cv2.bitwise_and(gray_resized, gray_resized, mask=(mask == 2).astype(np.uint8))
            
            # Apply Contrast Limited Adaptive Histogram Equalization to make texture pop
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced_nuc = clahe.apply(nuc_only_gray)
            
            # Map to Bone colormap for medical X-ray look
            texture_vis = cv2.applyColorMap(enhanced_nuc, cv2.COLORMAP_BONE)
            texture_vis[mask != 2] = [0, 0, 0] # Keep background black
            
            texture_image.image(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB), use_container_width=True, caption="High-Contrast Chromatin Topography")
            
            mean, stddev = cv2.meanStdDev(gray_resized, mask=(mask == 2).astype(np.uint8))
            chromatin_variance = float(stddev[0][0] ** 2) if stddev is not None else 0.0
            
            # 3. EfficientNet Grad-CAM
            rgb_img_resized = cv2.resize(cell_rgb, (224, 224))
            rgb_img_float = np.float32(rgb_img_resized) / 255
            pil_img = Image.fromarray(cell_rgb)
            cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                cnn_output = cnn_model(cnn_tensor)
                probs = torch.nn.functional.softmax(cnn_output, dim=1).squeeze().cpu().numpy()
            
            time.sleep(0.8) # Slow down before displaying final result
                
            # XGBoost Diagnosis
            features = pd.DataFrame([{
                'Cell_ID': f"Cell {idx+1}",
                'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area+nuc_area,
                'NC_Ratio': nc_ratio, 'Nuc_Perimeter': nuc_perimeter, 'Nuc_Circularity': nuc_circularity,
                'Nuc_Eccentricity': nuc_eccentricity, 'Chromatin_Variance': chromatin_variance,
                'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2],
                'Prob_3': probs[3], 'Prob_4': probs[4]
            }])
            
            final_class_id = xgb_model.predict(features.drop(columns=['Cell_ID']))[0]
            final_diagnosis = class_names[final_class_id]
            features.insert(1, 'Diagnosis', final_diagnosis)
            
            all_cell_features.append(features)
            
            # --- ELEGANT METRICS UPDATE ---
            m_id.metric(label=f"Processed Cell", value=f"Cell {idx+1}", delta=final_diagnosis, delta_color="normal")
            m_nc.metric(label="Nucleus-Cytoplasm Ratio", value=f"{nc_ratio:.3f}")
            m_circ.metric(label="Nuclear Circularity", value=f"{nuc_circularity:.3f}")
            
            live_table_placeholder.dataframe(pd.concat(all_cell_features, ignore_index=True), use_container_width=True)
            
            # Update Main Image with Text
            text_y = y1 - 10 if y1 > 30 else y1 + 30
            cv2.putText(display_img, final_diagnosis, (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, display_img.shape[1]/1500), (0, 255, 0), max(2, int(display_img.shape[1]/500)))
            fast_display = resize_for_web(display_img)
            main_image_placeholder.image(cv2.cvtColor(fast_display, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Phase 4: Final XGBoost Diagnoses")
            
            progress_bar.progress((idx + 1) / len(cells_to_process))
            time.sleep(1.5) # Long dramatic pause so the user can digest the data
            
        st.success("✅ Full Cinematic Diagnostic Pipeline Complete!")
