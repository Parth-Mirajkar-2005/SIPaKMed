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

st.set_page_config(page_title="SIPaKMeD Presentation Mode", layout="centered")

# --- APP_5: PRESENTATION / FOCUS UI ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #f0f2f6;
    }
    .huge-title {
        font-size: 3rem !important;
        text-align: center;
        font-weight: 800;
        color: #111;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.5rem !important;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .focus-diagnosis {
        font-size: 4rem;
        text-align: center;
        font-weight: 900;
        color: #e74c3c;
        margin-top: 20px;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
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

st.markdown("<p class='huge-title'>Focus Mode</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>One cell at a time. Maximum attention.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Insert Slide", type=["bmp", "jpg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    main_stage = st.empty()
    main_stage.image(cv2.cvtColor(resize_for_web(original_img), cv2.COLOR_BGR2RGB), use_container_width=True)
    
    if st.button("Begin Presentation", use_container_width=True, type="primary"):
        main_stage.empty() # Clear the big image
        
        # 1. Global Scan Animation
        scan_msg = st.empty()
        scan_msg.markdown("<p class='subtitle'>Scanning for targets...</p>", unsafe_allow_html=True)
        results = yolo_model(original_img, verbose=False)[0]
        
        cells_to_process = []
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                cells_to_process.append((int(x1), int(y1), int(x2), int(y2)))
        
        scan_msg.empty()
        
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        # 2. Present each cell one by one
        for idx, (x1, y1, x2, y2) in enumerate(cells_to_process):
            cell_crop = original_img[y1:y2, x1:x2]
            if cell_crop.size == 0: continue
            
            cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
            
            stage_header = st.empty()
            stage_header.markdown(f"<p class='subtitle'>Subject {idx+1} / {len(cells_to_process)}</p>", unsafe_allow_html=True)
            
            # Step 1: Show Raw Cell
            main_stage.image(resize_for_web(cell_rgb, 600), caption="Extracted Morphology", use_container_width=True)
            time.sleep(1.5)
            
            # Step 2: Show U-Net Mask overlaying it
            unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
            unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
            with torch.no_grad():
                mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                
            visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
            visual_mask[mask == 1] = [50, 200, 50]
            visual_mask[mask == 2] = [200, 50, 50]
            
            main_stage.image(resize_for_web(visual_mask, 600), caption="AI Segmentation Map", use_container_width=True)
            time.sleep(1.5)
            
            # Feature extraction for prediction
            cyt_area = float(np.sum(mask == 1))
            nuc_area = float(np.sum(mask == 2))
            nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
            
            pil_img = Image.fromarray(cell_rgb)
            cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
                
            features = pd.DataFrame([{
                'Target': "tmp", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
                'NC_Ratio': nc_ratio, 'Nuc_Perimeter': 0.0, 'Nuc_Circularity': 0.0,
                'Nuc_Eccentricity': 0.0, 'Chromatin_Variance': 0.0,
                'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
            }])
            final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
            diagnosis = class_names[final_class_id]
            
            # Step 3: Flash Diagnosis
            main_stage.empty()
            stage_header.empty()
            st.markdown(f"<div class='focus-diagnosis'>{diagnosis.upper()}</div>", unsafe_allow_html=True)
            time.sleep(2.5)
            
            st.experimental_rerun() if idx < len(cells_to_process) - 1 else None
            # Need to clear the previous diagnosis div by putting something empty, but streamlits markdown doesn't have an empty() equivalent easily, so we just use another empty container.
            
        st.success("Presentation complete.")
