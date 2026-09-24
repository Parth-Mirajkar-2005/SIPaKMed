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

st.set_page_config(page_title="SIPaKMeD Cinematic Deep-Dive", layout="wide")

# --- APP_10: CINEMATIC DEEP-DIVE BAY ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #000000 !important;
    }
    h1, h2, h3 {
        color: #f1f2f6 !important;
        font-family: 'Inter', sans-serif;
        text-align: center;
        letter-spacing: 2px;
    }
    .main-stage-container {
        display: flex;
        justify-content: center;
        margin-bottom: 40px;
        padding-top: 20px;
        border-bottom: 1px solid #333;
        padding-bottom: 40px;
    }
    .cinematic-lens {
        border-radius: 20px;
        box-shadow: 0 0 50px rgba(255, 255, 255, 0.1);
        border: 2px solid #333;
    }
    .deep-dive-card {
        background-color: #111;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        color: #ccc;
        text-align: center;
    }
    .deep-dive-card b {
        color: #fff;
    }
    .diagnosis-alert {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 10px -10px rgba(255,0,0,0); }
        to { box-shadow: 0 0 20px 5px rgba(231, 76, 60, 0.4); }
    }
</style>
""", unsafe_allow_html=True)

def resize_for_web(img, max_width=1000):
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

st.title("THE DEEP-DIVE BAY")
uploaded_file = st.file_uploader("UPLOAD GLOBAL SLIDE", type=["bmp", "jpg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    st.markdown("<div class='main-stage-container'>", unsafe_allow_html=True)
    stage = st.empty()
    stage.image(cv2.cvtColor(resize_for_web(original_img), cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("ENGAGE ANALYSIS SEQUENCE", use_container_width=True):
        results = yolo_model(original_img, verbose=False)[0]
        cells_to_process = []
        display_img = original_img.copy()
        
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cells_to_process.append((x1, y1, x2, y2, original_img[y1:y2, x1:x2]))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            stage.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
            time.sleep(1)
            
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        deep_dive_area = st.container()
        
        with deep_dive_area:
            st.markdown("### 🔬 ISOLATED TARGET ANALYSIS")
            for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                if cell_crop.size == 0: continue
                
                # Global Context Update
                focus_img = display_img.copy()
                cv2.rectangle(focus_img, (x1, y1), (x2, y2), (255, 0, 0), 6)
                stage.image(cv2.cvtColor(resize_for_web(focus_img), cv2.COLOR_BGR2RGB), use_container_width=True)
                
                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                
                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                    
                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [100, 100, 100]
                visual_mask[mask == 2] = [200, 0, 0]
                
                # Texture
                gray_img = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_img, (128, 128))
                nuc_only = cv2.bitwise_and(gray_resized, gray_resized, mask=(mask == 2).astype(np.uint8))
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced_nuc = clahe.apply(nuc_only)
                texture_vis = cv2.applyColorMap(enhanced_nuc, cv2.COLORMAP_INFERNO)
                texture_vis[mask != 2] = [0, 0, 0]
                
                cyt_area = float(np.sum(mask == 1))
                nuc_area = float(np.sum(mask == 2))
                nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
                
                pil_img = Image.fromarray(cell_rgb)
                cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
                    
                features = pd.DataFrame([{
                    'Target': f"tmp", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
                    'NC_Ratio': nc_ratio, 'Nuc_Perimeter': 0.0, 'Nuc_Circularity': 0.0,
                    'Nuc_Eccentricity': 0.0, 'Chromatin_Variance': 0.0,
                    'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
                }])
                final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
                diagnosis = class_names[final_class_id]
                
                is_danger = diagnosis in ['Dyskeratotic', 'Koilocytotic', 'Parabasal']
                bg_color = "#4a0e0e" if is_danger else "#0e4a19"
                text_color = "#ff7675" if is_danger else "#55efc4"
                
                c1, c2, c3 = st.columns(3)
                with c1: st.image(cell_rgb, caption="Raw Morphology", use_container_width=True)
                with c2: st.image(visual_mask, caption="U-Net Geometry", use_container_width=True)
                with c3: st.image(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB), caption="Chromatin Texture", use_container_width=True)
                
                st.markdown(f"""
                <div class='diagnosis-alert' style='background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};'>
                    {diagnosis.upper()} | NC_RATIO: {nc_ratio:.3f} | CONF: {np.max(probs)*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
                time.sleep(1.2)
                
        # Reset stage
        stage.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
