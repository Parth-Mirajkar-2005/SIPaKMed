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
import sys

st.set_page_config(page_title="SIPaKMeD Neural Net Terminal", layout="wide", page_icon="💻")

# --- APP_3: HACKER / DEVELOPER TERMINAL UI ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #0d1117;
    }
    * {
        font-family: 'Courier New', Courier, monospace !important;
        color: #00ff00 !important;
    }
    .terminal-header {
        border-bottom: 2px dashed #00ff00;
        padding-bottom: 10px;
        margin-bottom: 20px;
        text-shadow: 0 0 5px #00ff00;
    }
    .terminal-log {
        background-color: #000000;
        border: 1px solid #00ff00;
        padding: 15px;
        height: 300px;
        overflow-y: scroll;
        font-size: 0.85rem;
    }
    div.stButton > button:first-child {
        background-color: #000000;
        border: 1px solid #00ff00;
        color: #00ff00;
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #00ff00;
        color: #000000;
    }
    .stImage > img {
        border: 1px solid #00ff00;
        filter: grayscale(100%) contrast(150%) brightness(80%);
        transition: filter 0.5s;
    }
    .stImage > img:hover {
        filter: none;
    }
</style>
""", unsafe_allow_html=True)

def resize_for_web(img, max_width=600):
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

st.markdown("<h2 class='terminal-header'>SYSTEM.ROOT.SIPaKMeD_NET_V2.0</h2>", unsafe_allow_html=True)

col_term, col_vis = st.columns([1, 1.5])

with col_term:
    st.markdown("### >_ terminal_log")
    log_placeholder = st.empty()
    logs = ["> SYSTEM INITIALIZED", "> WAITING FOR IMAGE UPLOAD..."]
    log_placeholder.code("\\n".join(logs), language="bash")
    
    uploaded_file = st.file_uploader("UPLOAD TARGET [.bmp/.jpg]", type=["bmp", "jpg", "png"])
    execute = st.button("EXECUTE ./run_inference.sh")

def add_log(msg):
    logs.append(f"> {msg}")
    if len(logs) > 15:
        logs.pop(0)
    log_placeholder.code("\\n".join(logs), language="bash")
    time.sleep(0.3)

with col_vis:
    st.markdown("### >_ visual_buffer")
    vis_placeholder = st.empty()
    vis_placeholder.markdown("*NO SIGNAL*")

if uploaded_file and execute:
    with st.spinner(""):
        try:
            yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
            add_log(f"MODELS MOUNTED ON {str(device).upper()}")
        except Exception as e:
            add_log(f"FATAL ERROR: {e}")
            st.stop()
            
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    add_log(f"IMG_DECODED: SHAPE={original_img.shape} TYPE={original_img.dtype}")
    
    vis_placeholder.image(cv2.cvtColor(resize_for_web(original_img), cv2.COLOR_BGR2RGB), use_container_width=True)
    
    add_log("EXECUTING YOLOv8_DETECTION()...")
    results = yolo_model(original_img, verbose=False)[0]
    cells_to_process = []
    
    if len(results.boxes) == 0:
        add_log("WARNING: 0 TARGETS ACQUIRED. FALLBACK -> ENTIRE_IMG")
        cells_to_process.append((0, 0, original_img.shape[1], original_img.shape[0], original_img))
    else:
        add_log(f"TARGETS ACQUIRED: {len(results.boxes)}")
        display_img = original_img.copy()
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, _ = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cell_crop = original_img[y1:y2, x1:x2]
            cells_to_process.append((x1, y1, x2, y2, cell_crop))
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        vis_placeholder.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(1)
        
    add_log("ALLOCATING MEMORY FOR SEGMENTATION/CLASSIFICATION PIPELINE")
    
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
    
    for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
        add_log(f"--- PROCESSING TARGET_{idx} [{x1},{y1} -> {x2},{y2}] ---")
        cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
        
        # Show specific cell
        vis_placeholder.image(cv2.cvtColor(resize_for_web(cell_crop, 300), cv2.COLOR_BGR2RGB), caption=f"TARGET_{idx}")
        
        # U-Net
        unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
        unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
        add_log("FORWARD_PASS: UNET(TARGET)...")
        with torch.no_grad():
            mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
            
        cyt_area = float(np.sum(mask == 1))
        nuc_area = float(np.sum(mask == 2))
        add_log(f"U-NET_OUT: NUC_PX={nuc_area} CYT_PX={cyt_area}")
        
        if nuc_area < 5:
            add_log(f"ERR_TARGET_{idx}: NUC_AREA TOO SMALL. ABORTING.")
            continue
            
        nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
        
        # EfficientNet
        add_log("FORWARD_PASS: EFFICIENTNET(TARGET)...")
        pil_img = Image.fromarray(cell_rgb)
        cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
            
        add_log(f"CNN_OUT: MAX_PROB={np.max(probs):.4f}")
        
        # XGBoost
        add_log("INFERENCING XGBOOST_META_MODEL...")
        features = pd.DataFrame([{
            'Target': f"Cell {idx+1}", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
            'NC_Ratio': nc_ratio, 'Nuc_Perimeter': 0.0, 'Nuc_Circularity': 0.0,
            'Nuc_Eccentricity': 0.0, 'Chromatin_Variance': 0.0,
            'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
        }])
        final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
        final_diagnosis = class_names[final_class_id]
        add_log(f"XGB_PREDICT -> [ {final_diagnosis.upper()} ]")
        time.sleep(0.5)
        
    add_log("PROCESS_TERMINATED. EOP.")
    add_log(">>> WAITING FOR NEXT BATCH_")
