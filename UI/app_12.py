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
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="SIPaKMeD AR Command Pro", layout="wide", page_icon="🏥")

# --- APP_12: CLINICAL AR COMMAND CENTER + PRO EXTENSIONS ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #0f172a !important;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        background-color: #1e293b;
        padding: 15px 25px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-family: 'Orbitron', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        border-radius: 5px;
    }
    /* AR Lens styling inside tabs */
    .ar-lens-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    .hud-stat-box {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #38bdf8;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .hud-danger {
        border-left: 4px solid #ef4444;
    }
    hr {
        border-color: #334155;
    }
    /* Graph and Row Animation */
    .stPlotlyChart {
        animation: fadeInScale 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    @keyframes fadeInScale {
        0% { opacity: 0; transform: scaleX(0.9) translateX(-20px); }
        100% { opacity: 1; transform: scaleX(1) translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

def resize_for_web(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img

def apply_lens_effect(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    center_h, center_w = h // 2, w // 2
    lens_img = img[center_h - min_dim//2 : center_h + min_dim//2, center_w - min_dim//2 : center_w + min_dim//2]
    rgb_img = cv2.cvtColor(lens_img, cv2.COLOR_BGR2RGB)
    overlay = np.full_like(rgb_img, (0, 30, 60))
    return cv2.addWeighted(rgb_img, 0.85, overlay, 0.15, 0), lens_img

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

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209074.png", width=60)
    st.markdown("### Clinical AI Workspace")
    patient_id = st.text_input("Patient ID", value="PT-99X")
    uploaded_file = st.file_uploader("Upload Slide Image", type=["bmp", "jpg", "png"])
    run_btn = st.button("Start Analysis", type="primary", use_container_width=True)

st.title("🔬 Clinical Diagnostics AI")

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    scifi_lens, raw_square = apply_lens_effect(original_img)
    
    tab1, tab2, tab3 = st.tabs(["Target Scanner", "Cellular Analysis", "Diagnostic Report"])
    
    with tab1:
        st.markdown("<div class='ar-lens-container'>", unsafe_allow_html=True)
        lens_placeholder = st.empty()
        lens_placeholder.image(resize_for_web(scifi_lens, 800), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    if run_btn:
        with tab1:
            st.info("Analyzing Image...")
            results = yolo_model(raw_square, verbose=False)[0]
            cells_to_process = []
            display_lens = scifi_lens.copy()
            
            if len(results.boxes) > 0:
                for box in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, _ = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cells_to_process.append((x1, y1, x2, y2, raw_square[y1:y2, x1:x2]))
                    cv2.rectangle(display_lens, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    lens_placeholder.image(resize_for_web(display_lens, 800), use_container_width=True)
                    time.sleep(0.3)
                    
                st.success(f"Cells Detected: {len(cells_to_process)}")
                time.sleep(1)

        with tab2:
            st.markdown("### 🧬 Detailed Cellular Metrics")
            cnn_transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
            
            all_features = []
            
            for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                if cell_crop.size == 0: continue
                
                # Update AR lens focus in tab1
                focus_lens = display_lens.copy()
                cv2.rectangle(focus_lens, (x1, y1), (x2, y2), (255, 0, 100), 5)
                lens_placeholder.image(resize_for_web(focus_lens, 800), use_container_width=True)
                
                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                
                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                    
                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [56, 189, 248] # Cytoplasm light blue
                visual_mask[mask == 2] = [3, 105, 161] # Nucleus dark blue
                
                # --- NEW TEXTURE LOGIC ---
                gray_img = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_img, (128, 128))
                cell_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
                cell_only = cv2.bitwise_and(gray_resized, gray_resized, mask=cell_mask)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced_cell = clahe.apply(cell_only)
                texture_vis = cv2.applyColorMap(enhanced_cell, cv2.COLORMAP_INFERNO)
                texture_vis[cell_mask == 0] = [0, 0, 0]
                
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
                
                if nuc_area < 5:
                    diagnosis = "ANOMALY (NO NUCLEUS)"
                    
                features.insert(1, 'Diagnosis', diagnosis)
                all_features.append(features)
                
                is_danger = diagnosis in ['Dyskeratotic', 'Koilocytotic', 'Parabasal', 'ANOMALY (NO NUCLEUS)']
                hud_class = "hud-stat-box hud-danger" if is_danger else "hud-stat-box"
                
                # Render deep dive grid
                st.markdown(f"#### Cell ID: {idx+1}")
                img_col1, img_col2, img_col3, plot_col = st.columns([1, 1, 1, 2])
                with img_col1: 
                    st.image(cell_rgb, caption="Raw Image")
                with img_col2: 
                    st.image(visual_mask, caption="U-Net Mask")
                with img_col3: 
                    # --- ADDED TEXTURE IMAGE ---
                    st.image(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB), caption="Chromatin Texture")
                    
                with plot_col:
                    # --- ADDED PROBABILITY PLOT ---
                    prob_df = pd.DataFrame({'Class': class_names, 'Confidence (%)': probs * 100})
                    fig = px.bar(prob_df, x='Confidence (%)', y='Class', orientation='h',
                                 color='Confidence (%)', color_continuous_scale='Reds' if is_danger else 'Blues')
                    fig.update_layout(
                        title="EfficientNet Texture Probabilities",
                        height=200, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class='{hud_class}'>
                    <b>DIAGNOSIS:</b> {diagnosis.upper()} | <b>NC_RATIO:</b> {nc_ratio:.3f} | <b>PEAK_CONFIDENCE:</b> {np.max(probs)*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Auto-scroll injection
                components.html(
                    "<script>window.parent.document.querySelector('.main').scrollTo({top: window.parent.document.querySelector('.main').scrollHeight, behavior: 'smooth'});</script>",
                    height=0
                )
                
                time.sleep(0.8)
                
        with tab3:
            st.markdown("### 📋 Final Diagnostic Report")
            if all_features:
                df = pd.concat(all_features, ignore_index=True)
                st.markdown(f"**Patient:** {patient_id} | **Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}")
                abnormals = len(df[df['Diagnosis'].isin(['Dyskeratotic', 'Koilocytotic', 'Parabasal', 'ANOMALY (NO NUCLEUS)'])])
                
                m1, m2 = st.columns(2)
                m1.metric("TOTAL CELLS ANALYZED", len(df))
                
                delta_text = "CRITICAL" if abnormals > 0 else "NOMINAL"
                delta_color = "#ff4b4b" if abnormals > 0 else "#09ab3b"
                delta_bg = "rgba(255, 75, 75, 0.15)" if abnormals > 0 else "rgba(9, 171, 59, 0.15)"
                
                m2.markdown(f"""
                <div style="display: flex; flex-direction: column;">
                    <span style="font-size: 14px; color: #94a3b8; font-family: sans-serif;">ANOMALIES FLAGGED</span>
                    <div style="display: flex; align-items: center; gap: 12px; margin-top: 5px;">
                        <span style="color: {delta_color}; background-color: {delta_bg}; font-size: 14px; padding: 4px 10px; border-radius: 5px; font-weight: bold;">{delta_text}</span>
                        <span style="font-size: 36px; color: white; font-weight: bold;">{abnormals}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                def highlight_danger(row):
                    if row['Diagnosis'] in ['Dyskeratotic', 'Koilocytotic', 'Parabasal', 'ANOMALY (NO NUCLEUS)']:
                        return ['background-color: #5c1616; color: white'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(df.style.apply(highlight_danger, axis=1), use_container_width=True)
