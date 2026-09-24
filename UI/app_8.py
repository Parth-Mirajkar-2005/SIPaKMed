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
import plotly.graph_objects as go

st.set_page_config(page_title="SIPaKMeD Data Radar", layout="wide")

# --- APP_8: LIVE SCATTER-PLOT RADAR (DATA-DRIVEN) ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #1a1a2e !important;
        color: #e94560 !important;
    }
    h1, h2, h3 {
        color: #0f3460 !important;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .metrics-box {
        background-color: #16213e;
        border: 2px solid #e94560;
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-top: 20px;
    }
    .metrics-box img {
        border-radius: 50%;
        border: 3px solid #e94560;
        margin-bottom: 10px;
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

st.title("🛰️ SIPaKMeD Data Radar")
st.write("Live mapping of cellular coordinates based on AI-extracted morphological metrics.")

col_plot, col_cell = st.columns([2, 1])

with col_plot:
    uploaded_file = st.file_uploader("Initialize Radar [Upload Image]", type=["bmp", "jpg", "png"])
    plot_placeholder = st.empty()
    
with col_cell:
    st.markdown("### TARGET FEED")
    cell_placeholder = st.empty()

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    # Empty plot setup
    fig = go.Figure()
    fig.update_layout(
        title="Live Cellular Distribution (NC Ratio vs Texture Confidence)",
        xaxis_title="Nucleus-Cytoplasm Ratio",
        yaxis_title="AI Anomaly Confidence (%)",
        template="plotly_dark",
        plot_bgcolor="#16213e",
        paper_bgcolor="#1a1a2e",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 100])
    )
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    if st.button("PING RADAR"):
        with st.spinner("Extracting coordinates..."):
            results = yolo_model(original_img, verbose=False)[0]
            cells_to_process = []
            if len(results.boxes) > 0:
                for box in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, _ = box
                    cells_to_process.append(original_img[int(y1):int(y2), int(x1):int(x2)])
        
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        data_points = {'x': [], 'y': [], 'color': [], 'text': []}
        
        for idx, cell_crop in enumerate(cells_to_process):
            if cell_crop.size == 0: continue
            
            cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
            
            # U-Net
            unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
            unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
            with torch.no_grad():
                mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                
            cyt_area = float(np.sum(mask == 1))
            nuc_area = float(np.sum(mask == 2))
            nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0
            
            # CNN
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
            
            # Calculate Anomaly Confidence (Sum of probabilities for abnormal classes)
            # Assuming Dyskeratotic, Koilocytotic, Parabasal are anomalies
            anomaly_conf = (probs[0] + probs[1] + probs[3]) * 100
            
            is_danger = diagnosis in ['Dyskeratotic', 'Koilocytotic', 'Parabasal']
            color = '#e94560' if is_danger else '#0f3460'
            
            # Update plot data
            data_points['x'].append(nc_ratio)
            data_points['y'].append(anomaly_conf)
            data_points['color'].append(color)
            data_points['text'].append(f"ID: {idx} | {diagnosis}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_points['x'],
                y=data_points['y'],
                mode='markers+text',
                marker=dict(size=15, color=data_points['color'], line=dict(width=2, color='white')),
                text=data_points['text'],
                textposition="top center"
            ))
            fig.update_layout(
                title="Live Cellular Distribution",
                xaxis_title="Nucleus-Cytoplasm Ratio",
                yaxis_title="AI Anomaly Confidence (%)",
                template="plotly_dark",
                plot_bgcolor="#16213e",
                paper_bgcolor="#1a1a2e",
                xaxis=dict(range=[0, max(1, max(data_points['x']) + 0.2) if data_points['x'] else 1]),
                yaxis=dict(range=[0, 105])
            )
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Show live cell side-feed
            with cell_placeholder:
                st.markdown(f"""
                <div class='metrics-box'>
                    <h3 style='color: white !important;'>TARGET {idx}</h3>
                    <b>Class:</b> <span style='color: {color};'>{diagnosis}</span><br>
                    <b>NC Ratio:</b> {nc_ratio:.3f}<br>
                    <b>Anomaly Score:</b> {anomaly_conf:.1f}%
                </div>
                """, unsafe_allow_html=True)
                st.image(cell_rgb, use_container_width=True)
            
            time.sleep(0.8)
