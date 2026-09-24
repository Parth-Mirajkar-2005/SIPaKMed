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

st.set_page_config(page_title="SIPaKMeD AR Lens", layout="wide")

# --- APP_6: SCI-FI MICROSCOPE LENS ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #050a15 !important;
    }
    h1, h2, h3, p {
        color: #00ffff !important;
        font-family: 'Orbitron', 'Courier New', sans-serif !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    /* Circular lens effect for main image */
    .lens-view {
        display: block;
        margin: 0 auto;
        border-radius: 50% !important;
        border: 4px solid #00ffff;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6), inset 0 0 30px rgba(0, 255, 255, 0.6);
        max-width: 600px;
        max-height: 600px;
        object-fit: cover;
    }
    /* HUD floating cards */
    .hud-card {
        background: rgba(0, 20, 40, 0.8);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
        color: #fff !important;
        font-family: monospace;
        backdrop-filter: blur(5px);
        animation: pulseBorder 2s infinite;
    }
    .hud-card b {
        color: #00ffff;
    }
    .hud-danger {
        border: 1px solid #ff0055;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.4);
    }
    .hud-danger b {
        color: #ff0055;
    }
    @keyframes pulseBorder {
        0% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.2); }
        50% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.6); }
        100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.2); }
    }
    div.stButton > button {
        background-color: transparent !important;
        border: 2px solid #00ffff !important;
        color: #00ffff !important;
        border-radius: 20px !important;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #00ffff !important;
        color: #000 !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.8) !important;
    }
    /* Apply lens styling to streamlits image component if we can't inject raw HTML easily */
    [data-testid="stImage"] img {
        border: 2px solid #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        border-radius: 15px;
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

st.title("👁️ AUGMENTED REALITY CYTOLOGY LENS")
st.markdown("### OPTICAL SCANNER ACTIVE")

col_lens, col_hud = st.columns([1.5, 1])

with col_lens:
    uploaded_file = st.file_uploader("MOUNT SPECIMEN SLIDE", type=["bmp", "jpg", "png"])
    main_lens = st.empty()

with col_hud:
    hud_panel = st.container()

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    # Square crop the image for the "lens" effect
    h, w = original_img.shape[:2]
    min_dim = min(h, w)
    center_h, center_w = h // 2, w // 2
    lens_img = original_img[center_h - min_dim//2 : center_h + min_dim//2, center_w - min_dim//2 : center_w + min_dim//2]
    
    # Convert to RGB and apply a slight bluish tint for the sci-fi feel
    rgb_img = cv2.cvtColor(lens_img, cv2.COLOR_BGR2RGB)
    overlay = np.full_like(rgb_img, (0, 50, 100))
    scifi_lens = cv2.addWeighted(rgb_img, 0.8, overlay, 0.2, 0)
    
    main_lens.image(resize_for_web(scifi_lens, 800), use_container_width=True)
    
    if st.button("ACTIVATE TARGET LOCK", use_container_width=True):
        with hud_panel:
            st.markdown("### TARGET ACQUISITION...")
            
        results = yolo_model(lens_img, verbose=False)[0]
        cells_to_process = []
        
        display_lens = scifi_lens.copy()
        
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cell_crop = lens_img[y1:y2, x1:x2]
                cells_to_process.append((x1, y1, x2, y2, cell_crop))
                # Draw neon cyan targeting brackets
                cv2.rectangle(display_lens, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
            main_lens.image(resize_for_web(display_lens, 800), use_container_width=True)
            time.sleep(1)
            
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        with hud_panel:
            st.markdown("### BIO-METRIC HUD")
            scroll_hud = st.container()
            with scroll_hud:
                for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                    if cell_crop.size == 0: continue
                    cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                    
                    # Highlight current target in lens
                    focus_lens = display_lens.copy()
                    cv2.rectangle(focus_lens, (x1, y1), (x2, y2), (255, 0, 85), 5) # Neon pink focus
                    main_lens.image(resize_for_web(focus_lens, 800), use_container_width=True)
                    
                    # U-Net
                    unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                    unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                    with torch.no_grad():
                        mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                        
                    visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                    visual_mask[mask == 1] = [0, 100, 100] # Dim cyan
                    visual_mask[mask == 2] = [0, 255, 255] # Bright cyan
                    
                    cyt_area = float(np.sum(mask == 1))
                    nuc_area = float(np.sum(mask == 2))
                    nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
                    
                    # EfficientNet
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
                    
                    # Check if danger class
                    is_danger = diagnosis in ['Dyskeratotic', 'Koilocytotic', 'Parabasal']
                    hud_class = "hud-card hud-danger" if is_danger else "hud-card"
                    
                    col_img, col_data = st.columns([1, 1.5])
                    with col_img:
                        st.image(visual_mask, caption=f"TGT-{idx+1}")
                    with col_data:
                        st.markdown(f"""
                        <div class='{hud_class}'>
                            <b>ID:</b> TGT-{idx+1}<br>
                            <b>NC_RATIO:</b> {nc_ratio:.2f}<br>
                            <b>CONF:</b> {np.max(probs)*100:.1f}%<br>
                            <b>STAT:</b> {diagnosis.upper()}
                        </div>
                        """, unsafe_allow_html=True)
                    time.sleep(1)
                    
        # Reset lens
        main_lens.image(resize_for_web(display_lens, 800), use_container_width=True)
