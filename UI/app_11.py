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

st.set_page_config(page_title="SIPaKMeD Master", layout="wide")

# --- APP_11: ULTIMATE MASTER INTERFACE ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #0b0c10 !important;
        color: #c5c6c7 !important;
    }
    .top-bar {
        background-color: #1f2833;
        padding: 15px;
        border-bottom: 3px solid #66fcf1;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
    }
    .metric-top {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        color: #45a29e;
    }
    .ar-lens-container {
        border-radius: 50%;
        border: 4px solid #66fcf1;
        box-shadow: 0 0 20px #66fcf1;
        margin: 0 auto;
        display: block;
        max-width: 500px;
        max-height: 500px;
        object-fit: cover;
    }
    .cinematic-feed-card {
        background-color: #1f2833;
        border-left: 5px solid #66fcf1;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    .feed-danger {
        border-left: 5px solid #ff4a4a;
    }
    h1, h2, h3 { color: #66fcf1 !important; }
</style>
""", unsafe_allow_html=True)

def resize_for_web(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def get_lens(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    center_h, center_w = h // 2, w // 2
    return img[center_h - min_dim//2 : center_h + min_dim//2, center_w - min_dim//2 : center_w + min_dim//2]

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

# Top Bar (Clinical Dashboard style)
st.markdown(f"""
<div class='top-bar'>
    <div class='metric-top'><b>SIPaKMeD MASTER UI</b></div>
    <div class='metric-top'>SYS_DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    <div class='metric-top'>STATUS: STANDBY</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("MOUNT SPECIMEN", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    raw_lens = get_lens(original_img)
    scifi_lens = cv2.addWeighted(cv2.cvtColor(raw_lens, cv2.COLOR_BGR2RGB), 0.8, np.full_like(raw_lens, (0,30,50)), 0.2, 0)
    
    col_lens, col_feed = st.columns([1, 1.5])
    
    with col_lens:
        st.markdown("### GLOBAL AR SCANNER")
        lens_stage = st.empty()
        st.markdown("<div class='ar-lens-container'>", unsafe_allow_html=True)
        lens_stage.image(resize_for_web(scifi_lens, 600), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        init_btn = st.button("INITIALIZE", use_container_width=True)
        
    with col_feed:
        st.markdown("### CINEMATIC FEED")
        feed_stage = st.container()
        
    if init_btn:
        with col_lens:
            st.info("SCANNING...")
        results = yolo_model(raw_lens, verbose=False)[0]
        cells_to_process = []
        display_lens = scifi_lens.copy()
        
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cells_to_process.append((x1, y1, x2, y2, raw_lens[y1:y2, x1:x2]))
                cv2.rectangle(display_lens, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
            lens_stage.image(resize_for_web(display_lens, 600), use_container_width=True)
            time.sleep(1)
            
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        all_features = []
        
        p_bar = st.progress(0)
        
        with feed_stage:
            for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                if cell_crop.size == 0: continue
                
                # Lens Focus (AR Lens feature)
                focus_lens = display_lens.copy()
                cv2.rectangle(focus_lens, (x1, y1), (x2, y2), (255, 74, 74), 6)
                lens_stage.image(resize_for_web(focus_lens, 600), use_container_width=True)
                
                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                
                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                    
                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [102, 252, 241]
                visual_mask[mask == 2] = [69, 162, 158]
                
                cyt_area = float(np.sum(mask == 1))
                nuc_area = float(np.sum(mask == 2))
                nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
                
                pil_img = Image.fromarray(cell_rgb)
                cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
                    
                features = pd.DataFrame([{
                    'Target': f"TGT-{idx+1}", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
                    'NC_Ratio': nc_ratio, 'Nuc_Perimeter': 0.0, 'Nuc_Circularity': 0.0,
                    'Nuc_Eccentricity': 0.0, 'Chromatin_Variance': 0.0,
                    'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
                }])
                final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
                diagnosis = class_names[final_class_id]
                features.insert(1, 'Diagnosis', diagnosis)
                all_features.append(features)
                
                is_danger = diagnosis in ['Dyskeratotic', 'Koilocytotic', 'Parabasal']
                feed_class = "cinematic-feed-card feed-danger" if is_danger else "cinematic-feed-card"
                
                # Cinematic Layout (app_1 feature)
                st.markdown(f"<div class='{feed_class}'>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1, 1.5])
                with c1: st.image(cell_rgb, caption="Raw Morphology")
                with c2: st.image(visual_mask, caption="U-Net Mask")
                with c3:
                    st.markdown(f"""
                    <b>TARGET ID:</b> TGT-{idx+1}<br>
                    <b>DIAGNOSIS:</b> <span style='font-size: 1.2rem; color: {'#ff4a4a' if is_danger else '#66fcf1'};'>{diagnosis.upper()}</span><br>
                    <b>NC RATIO:</b> {nc_ratio:.3f}<br>
                    <b>CONFIDENCE:</b> {np.max(probs)*100:.1f}%
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                p_bar.progress((idx + 1) / len(cells_to_process))
                time.sleep(0.8)
                
        # Reset Lens
        lens_stage.image(resize_for_web(display_lens, 600), use_container_width=True)
        
        # Bottom Summary (Clinical Dashboard feature)
        st.divider()
        st.markdown("### CLINICAL SUMMARY")
        if all_features:
            df = pd.concat(all_features, ignore_index=True)
            abnormals = len(df[df['Diagnosis'].isin(['Dyskeratotic', 'Koilocytotic', 'Parabasal'])])
            m1, m2 = st.columns(2)
            m1.metric("TOTAL SCANNED", len(df))
            m2.metric("ANOMALIES", abnormals, delta="FLAGGED" if abnormals > 0 else "CLEAN", delta_color="inverse")
            st.dataframe(df, use_container_width=True)
