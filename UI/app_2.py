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
import datetime

st.set_page_config(page_title="SIPaKMeD Clinical Dashboard", layout="wide", page_icon="🏥")

# --- APP_2: CLINICAL DASHBOARD (LIGHT/PROFESSIONAL THEME) ---
st.markdown("""
<style>
    .report-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .patient-info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def resize_for_web(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

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

with st.spinner("Initializing Clinical AI Models..."):
    try:
        yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}. Please check file paths.")
        st.stop()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']

# --- SIDEBAR: CLINICAL CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209074.png", width=80)
    st.markdown("### Clinical Diagnostics Panel")
    patient_id = st.text_input("Patient ID", value="PT-8492-AX")
    technician = st.text_input("Technician Name", value="Dr. Smith")
    uploaded_file = st.file_uploader("Upload Cytology Slide", type=["bmp", "jpg", "png", "jpeg"])
    st.divider()
    run_btn = st.button("Generate Diagnostic Report", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
st.markdown("<h1 class='report-header'>Automated Cytopathology Report</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    st.markdown(f"""
    <div class='patient-info-box'>
        <strong>Patient ID:</strong> {patient_id} &nbsp;&nbsp;|&nbsp;&nbsp; 
        <strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; 
        <strong>Technician:</strong> {technician}
    </div>
    """, unsafe_allow_html=True)
    
    # Define Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Raw Slide", "🎯 Detection", "🔬 Segmentation & Texture", "📋 Final Report"])
    
    with tab1:
        st.image(cv2.cvtColor(resize_for_web(original_img), cv2.COLOR_BGR2RGB), use_container_width=True)
        
    if run_btn:
        # Phase 1: YOLO
        with tab2:
            st.info("Initiating Phase 1: Object Detection (YOLOv8)")
            yolo_placeholder = st.empty()
            yolo_metrics = st.empty()
            
            with st.spinner("Scanning for cellular anomalies..."):
                time.sleep(1) # Simulated loading
                results = yolo_model(original_img, verbose=False)[0]
                
            cells_to_process = []
            display_img = original_img.copy()
            
            if len(results.boxes) == 0:
                yolo_metrics.warning("No distinct cells detected. Entire field will be processed.")
                h, w, _ = original_img.shape
                cells_to_process.append((0, 0, w, h, original_img))
            else:
                yolo_metrics.success(f"Detected {len(results.boxes)} targets of interest.")
                for box in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, _ = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cell_crop = original_img[y1:y2, x1:x2]
                    if cell_crop.size > 0:
                        cells_to_process.append((x1, y1, x2, y2, cell_crop))
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), max(2, int(display_img.shape[1]/500)))
                    yolo_placeholder.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
                    time.sleep(0.3)
                    
        # Phase 2 & 3: Segmentation & Texture
        with tab3:
            st.info("Initiating Phase 2: Morphometric & Topographical Analysis")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Setup columns for the current cell being analyzed
            col_crop, col_mask, col_tex = st.columns(3)
            with col_crop:
                st.markdown("**Isolated Cell**")
                crop_img_ui = st.empty()
            with col_mask:
                st.markdown("**U-Net Segmentation**")
                mask_img_ui = st.empty()
            with col_tex:
                st.markdown("**Chromatin Map**")
                tex_img_ui = st.empty()
                
            all_cell_features = []
            
            for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                progress_text.text(f"Analyzing Target {idx+1} of {len(cells_to_process)}...")
                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                crop_img_ui.image(cell_rgb, use_container_width=True)
                
                # U-Net processing
                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                
                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [173, 216, 230] # Light Blue Cytoplasm
                visual_mask[mask == 2] = [0, 0, 139] # Dark Blue Nucleus
                mask_img_ui.image(visual_mask, use_container_width=True)
                
                cyt_area = float(np.sum(mask == 1))
                nuc_area = float(np.sum(mask == 2))
                nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
                
                if nuc_area < 5:
                    tex_img_ui.error("QC Failed: Insufficient Nucleus")
                    time.sleep(1)
                    continue
                
                nuc_perimeter, nuc_circularity, nuc_eccentricity = 0.0, 0.0, 0.0
                nuc_mask_uint8 = (mask == 2).astype(np.uint8) * 255
                contours, _ = cv2.findContours(nuc_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    nuc_perimeter = float(cv2.arcLength(largest_contour, True))
                    if nuc_perimeter > 0: nuc_circularity = 4 * np.pi * (nuc_area / (nuc_perimeter ** 2))
                    if len(largest_contour) >= 5:
                        ellipse = cv2.fitEllipse(largest_contour)
                        if ellipse[1][0] > 0: nuc_eccentricity = ellipse[1][1] / ellipse[1][0]
                
                # Texture
                gray_img = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_img, (128, 128))
                nuc_only_gray = cv2.bitwise_and(gray_resized, gray_resized, mask=(mask == 2).astype(np.uint8))
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                enhanced_nuc = clahe.apply(nuc_only_gray)
                texture_vis = cv2.applyColorMap(enhanced_nuc, cv2.COLORMAP_OCEAN)
                texture_vis[mask != 2] = [255, 255, 255]
                tex_img_ui.image(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                mean, stddev = cv2.meanStdDev(gray_resized, mask=(mask == 2).astype(np.uint8))
                chromatin_variance = float(stddev[0][0] ** 2) if stddev is not None else 0.0
                
                # EfficientNet
                rgb_img_float = np.float32(cv2.resize(cell_rgb, (224, 224))) / 255
                pil_img = Image.fromarray(cell_rgb)
                cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
                
                # XGBoost
                features = pd.DataFrame([{
                    'Target': f"Cell {idx+1}", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
                    'NC_Ratio': nc_ratio, 'Nuc_Perimeter': nuc_perimeter, 'Nuc_Circularity': nuc_circularity,
                    'Nuc_Eccentricity': nuc_eccentricity, 'Chromatin_Variance': chromatin_variance,
                    'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
                }])
                final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
                features.insert(1, 'Diagnosis', class_names[final_class_id])
                all_cell_features.append(features)
                
                progress_bar.progress((idx + 1) / len(cells_to_process))
                time.sleep(1) # Let user observe the dashboard

        # Phase 4: Final Report
        with tab4:
            if not all_cell_features:
                st.error("No valid cells processed.")
            else:
                st.success("Analysis Complete. Final diagnostic report generated.")
                df_final = pd.concat(all_cell_features, ignore_index=True)
                
                # Summary metrics
                abnormal_count = len(df_final[df_final['Diagnosis'].isin(['Dyskeratotic', 'Koilocytotic', 'Parabasal'])])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Cells Analyzed", len(df_final))
                c2.metric("Abnormal Cells Flagged", abnormal_count, delta="Requires Review" if abnormal_count > 0 else "Normal", delta_color="inverse")
                c3.metric("Dominant Morphology", df_final['Diagnosis'].mode()[0])
                
                st.markdown("### Detailed Feature Matrix")
                st.dataframe(df_final.style.highlight_max(axis=0), use_container_width=True)
                
                st.button("Export PDF Report", type="secondary")
else:
    st.info("Please upload a cytology slide from the sidebar to begin the analysis.")
