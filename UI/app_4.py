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

st.set_page_config(page_title="SIPaKMeD Analytical Split", layout="wide")

# --- APP_4: SIDE-BY-SIDE ANALYTICAL UI ---
st.markdown("""
<style>
    .split-header {
        text-align: center;
        padding: 10px;
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .data-card {
        border-left: 4px solid #e74c3c;
        background-color: #f8f9fa;
        padding: 10px;
        margin-bottom: 15px;
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

with st.spinner("Loading analytical models..."):
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()

st.markdown("<h2 class='split-header'>Contextual Micro-Analysis Protocol</h2>", unsafe_allow_html=True)
st.write("Upload an image. The left panel will maintain the global context, while the right panel dives deep into each detected cellular structure.")

uploaded_file = st.file_uploader("Upload slide image", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    display_img = original_img.copy()
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("Global Context View")
        context_placeholder = st.empty()
        context_placeholder.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col_right:
        st.subheader("Deep Dive Pipeline")
        right_panel = st.empty()
        
    if st.button("Initialize Contextual Scan", use_container_width=True):
        with col_right:
            st.info("Scanning global context for targets...")
            
        results = yolo_model(original_img, verbose=False)[0]
        cells_to_process = []
        
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cell_crop = original_img[y1:y2, x1:x2]
                cells_to_process.append((x1, y1, x2, y2, cell_crop))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
            # Update left context with bounding boxes
            context_placeholder.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
            time.sleep(0.5)
            
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        with col_right:
            scroll_container = st.container()
            with scroll_container:
                for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                    st.markdown("---")
                    st.markdown(f"### Target Location: [{x1}, {y1}]")
                    cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                    
                    # Temporarily highlight the current cell on the left context view
                    focus_img = display_img.copy()
                    cv2.rectangle(focus_img, (x1, y1), (x2, y2), (0, 0, 255), 4) # Red focus box
                    context_placeholder.image(cv2.cvtColor(resize_for_web(focus_img), cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(cell_rgb, caption="Raw Extract")
                        
                    # U-Net processing
                    unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                    unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                    with torch.no_grad():
                        mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                        
                    visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                    visual_mask[mask == 1] = [150, 150, 150]
                    visual_mask[mask == 2] = [255, 50, 50]
                    with c2:
                        st.image(visual_mask, caption="Morphology")
                        
                    cyt_area = float(np.sum(mask == 1))
                    nuc_area = float(np.sum(mask == 2))
                    nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
                    
                    # EfficientNet
                    pil_img = Image.fromarray(cell_rgb)
                    cnn_tensor = cnn_transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().cpu().numpy()
                        
                    # XGBoost
                    features = pd.DataFrame([{
                        'Target': "tmp", 'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': cyt_area + nuc_area,
                        'NC_Ratio': nc_ratio, 'Nuc_Perimeter': 0.0, 'Nuc_Circularity': 0.0,
                        'Nuc_Eccentricity': 0.0, 'Chromatin_Variance': 0.0,
                        'Prob_0': probs[0], 'Prob_1': probs[1], 'Prob_2': probs[2], 'Prob_3': probs[3], 'Prob_4': probs[4]
                    }])
                    final_class_id = xgb_model.predict(features.drop(columns=['Target']))[0]
                    diagnosis = class_names[final_class_id]
                    
                    with c3:
                        st.markdown(f"""
                        <div class='data-card'>
                            <b>NC Ratio:</b> {nc_ratio:.3f}<br>
                            <b>Max Prob:</b> {np.max(probs):.3f}<br>
                            <b>Class:</b> <span style='color: #e74c3c;'>{diagnosis}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    time.sleep(1) # Delay so user can trace the context
                    
        # Reset the left view back to normal bounding boxes after finishing
        context_placeholder.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.success("Contextual Analysis Complete.")
