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

st.set_page_config(page_title="SIPaKMeD Editorial", layout="centered")

# --- APP_7: STORYBOOK / EDITORIAL TIMELINE ---
st.markdown("""
<style>
    body, .stApp {
        background-color: #fbfbf9 !important; /* Soft paper white */
    }
    h1 {
        font-family: 'Georgia', serif !important;
        font-size: 3.5rem !important;
        color: #2c3e50 !important;
        text-align: center;
        border-bottom: 1px solid #dcdcdc;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    h2, h3 {
        font-family: 'Georgia', serif !important;
        color: #34495e !important;
        margin-top: 40px;
    }
    p, .stMarkdown {
        font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
        font-size: 1.15rem;
        color: #4a4a4a;
        line-height: 1.8;
    }
    .drop-cap:first-letter {
        float: left;
        font-size: 4rem;
        line-height: 0.8;
        padding-top: 4px;
        padding-right: 8px;
        padding-left: 3px;
        color: #e74c3c;
        font-family: 'Georgia', serif;
    }
    .chapter-divider {
        text-align: center;
        margin: 40px 0;
        font-size: 2rem;
        color: #bdc3c7;
    }
    .stExpander {
        border: none !important;
        border-bottom: 1px solid #eee !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    .stImage > img {
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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

st.title("The Cellular Narrative")

uploaded_file = st.file_uploader("Provide the Manuscript (Slide)", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, xgb_model, device = load_models()
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    st.markdown(f"<p class='drop-cap'>On this day, {datetime.datetime.now().strftime('%B %d, %Y')}, a new microscopic landscape was brought to our attention. The vast expanse of the slide laid before us, a complex tapestry of biological activity waiting to be deciphered. Our journey begins with a broad survey of the terrain.</p>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(resize_for_web(original_img), cv2.COLOR_BGR2RGB), caption="The initial, untouched landscape.", use_container_width=True)
    
    if st.button("Unfold the Narrative"):
        st.markdown("<div class='chapter-divider'>***</div>", unsafe_allow_html=True)
        st.markdown("## Chapter 1: The Discovery")
        st.write("Using the YOLOv8 architecture as our scout, we scanned the horizon for objects of interest. The neural network, trained on thousands of prior expeditions, quickly began placing markers upon structures that deviated from the background noise.")
        
        results = yolo_model(original_img, verbose=False)[0]
        cells_to_process = []
        display_img = original_img.copy()
        
        if len(results.boxes) > 0:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, _ = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cell_crop = original_img[y1:y2, x1:x2]
                cells_to_process.append((x1, y1, x2, y2, cell_crop))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
            st.image(cv2.cvtColor(resize_for_web(display_img), cv2.COLOR_BGR2RGB), caption=f"Our scout returned with {len(results.boxes)} distinct regions of interest.", use_container_width=True)
            time.sleep(1.5)
            
        st.markdown("<div class='chapter-divider'>***</div>", unsafe_allow_html=True)
        st.markdown("## Chapter 2: The Deep Dive")
        st.write("With the targets identified, we shifted our focus from the macro to the micro. For each isolated subject, we employed a U-Net model to map its internal geography—delineating the dark nucleus from the surrounding cytoplasm. This was followed by a textural examination via EfficientNet to capture the nuances of its chromatin.")
        
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        class_names = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        
        final_diagnoses = []
        
        for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
            if cell_crop.size == 0: continue
            
            with st.expander(f"Subject #{idx+1}: Coordinates [{x1}, {y1}]"):
                col_story1, col_story2 = st.columns(2)
                
                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
                
                with col_story1:
                    st.image(cell_rgb, caption="The Raw Extract")
                    
                # U-Net
                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()
                    
                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [230, 230, 230] # Light gray
                visual_mask[mask == 2] = [80, 80, 80] # Dark gray
                
                with col_story2:
                    st.image(visual_mask, caption="The Topographical Map")
                    
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
                final_diagnoses.append(diagnosis)
                
                st.write(f"The topographical analysis revealed a Nucleus-to-Cytoplasm ratio of **{nc_ratio:.3f}**. "
                         f"Combined with the textural probabilities, our Meta-Learner concluded that this structure exhibits characteristics consistent with **{diagnosis}** morphology.")
                time.sleep(0.5)

        st.markdown("<div class='chapter-divider'>***</div>", unsafe_allow_html=True)
        st.markdown("## Chapter 3: The Verdict")
        
        if final_diagnoses:
            dominant = max(set(final_diagnoses), key=final_diagnoses.count)
            abnormal = [d for d in final_diagnoses if d in ['Dyskeratotic', 'Koilocytotic', 'Parabasal']]
            
            st.write(f"After meticulously examining all {len(cells_to_process)} subjects, a pattern emerged. The overarching narrative of this tissue sample was defined by a predominantly **{dominant}** presence.")
            
            if len(abnormal) > 0:
                st.write(f"However, the story does not end there. We found {len(abnormal)} instances of anomalous cellular structures that demand further investigation by a human specialist.")
                st.error("Conclusion: Pathological Review Recommended.")
            else:
                st.write("No significant anomalies were detected during this expedition. The landscape remains serene.")
                st.success("Conclusion: Benign / Normal Range.")
                
        st.write("---")
        st.write("*Fin.*")
