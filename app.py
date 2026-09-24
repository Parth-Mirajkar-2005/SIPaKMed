"""
APP_13: Super Ensemble Clinical Diagnostics AI
===============================================
Based on app_12's proven UI design, upgraded to use the full
Super Ensemble (XGBoost + LightGBM + Random Forest majority vote)
with complete biological feature extraction.

Key optimization: ALL deep learning inference runs on CPU to avoid
CUDA OOM on 6GB GPUs. For single-image UI usage this is fast enough.
"""

import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import time
import datetime
import plotly.express as px
import joblib
import gc

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SIPaKMeD Super Ensemble AI",
    layout="wide",
    page_icon="🏥"
)

# ──────────────────────────────────────────────
# CUSTOM CSS (identical to app_12)
# ──────────────────────────────────────────────
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
    .stPlotlyChart {
        animation: fadeInScale 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    @keyframes fadeInScale {
        0% { opacity: 0; transform: scaleX(0.9) translateX(-20px); }
        100% { opacity: 1; transform: scaleX(1) translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SUPER ENSEMBLE CLASS (needed to unpickle .pkl)
# ──────────────────────────────────────────────
from scipy.stats import mode as scipy_mode

class CervicalCancerSuperEnsemble:
    """Hard Majority Vote across XGBoost, LightGBM, Random Forest."""
    def __init__(self, xgb, lgb, rf):
        self.xgb = xgb
        self.lgb = lgb
        self.rf = rf
        self.xgb_feats = list(xgb.feature_names_in_)
        self.lgb_feats = list(lgb.feature_names_in_)
        self.rf_feats = list(rf.feature_names_in_)

    def predict(self, X):
        p1 = self.xgb.predict(X[self.xgb_feats])
        p2 = self.lgb.predict(X[self.lgb_feats])
        p3 = self.rf.predict(X[self.rf_feats])
        stacked = np.stack([p1, p2, p3], axis=1)
        return scipy_mode(stacked, axis=1, keepdims=False).mode

    def predict_proba(self, X):
        p1 = self.xgb.predict_proba(X[self.xgb_feats])
        p2 = self.lgb.predict_proba(X[self.lgb_feats])
        p3 = self.rf.predict_proba(X[self.rf_feats])
        return (p1 + p2 + p3) / 3.0


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
CLASS_NAMES = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic',
               'Parabasal', 'Superficial-Intermediate']

DANGER_CLASSES = {'Dyskeratotic', 'Koilocytotic', 'Parabasal', 'ANOMALY (NO NUCLEUS)'}

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def resize_for_web(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(img, (max_width, int(h * ratio)))
    return img


def apply_lens_effect(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    ch, cw = h // 2, w // 2
    lens = img[ch - min_dim // 2: ch + min_dim // 2,
               cw - min_dim // 2: cw + min_dim // 2]
    rgb = cv2.cvtColor(lens, cv2.COLOR_BGR2RGB)
    overlay = np.full_like(rgb, (0, 30, 60))
    return cv2.addWeighted(rgb, 0.85, overlay, 0.15, 0), lens


def extract_features_for_cell(mask, cell_crop, cnn_probs):
    import scipy.stats as sts
    h_mask, w_mask = mask.shape
    cell_resized = cv2.resize(cell_crop, (w_mask, h_mask))

    nuc_mask = (mask == 2).astype(np.uint8) * 255
    cyto_mask = (mask == 1).astype(np.uint8) * 255

    nuc_area = float(np.sum(mask == 2))
    cyt_area = float(np.sum(mask == 1))
    total_area = cyt_area + nuc_area
    nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0.0

    nuc_perimeter = nuc_circularity = nuc_eccentricity = 0.0
    aspect = solidity = extent = centroid_dist = 0.0
    hu = [0.0] * 7
    fd_mag = [0.0] * 5

    contours, _ = cv2.findContours(nuc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        nuc_perimeter = float(cv2.arcLength(cnt, True))
        if nuc_perimeter > 0:
            nuc_circularity = 4 * np.pi * nuc_area / (nuc_perimeter ** 2)
        if len(cnt) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            if MA > 0:
                nuc_eccentricity = ma / MA
                aspect = MA / ma
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = nuc_area / hull_area
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 0:
            extent = nuc_area / (w * h)

        hu = cv2.HuMoments(cv2.moments(cnt)).flatten().tolist()

        M_n = cv2.moments(cnt)
        if M_n['m00'] != 0:
            cx_n = M_n['m10'] / M_n['m00']
            cy_n = M_n['m01'] / M_n['m00']
            c_cnts, _ = cv2.findContours(cyto_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if c_cnts:
                c_cnt = max(c_cnts, key=cv2.contourArea)
                M_c = cv2.moments(c_cnt)
                if M_c['m00'] != 0:
                    centroid_dist = np.sqrt(
                        (cx_n - M_c['m10'] / M_c['m00']) ** 2 +
                        (cy_n - M_c['m01'] / M_c['m00']) ** 2
                    )

        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 0:
            fd = np.fft.fft(pts[:, 0] + 1j * pts[:, 1])
            fd_mag = np.abs(fd[:5]).tolist()
            while len(fd_mag) < 5:
                fd_mag.append(0.0)

    gray = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)
    gray_nuc = cv2.bitwise_and(gray, gray, mask=nuc_mask)
    _, stddev = cv2.meanStdDev(gray, mask=nuc_mask)
    chromatin_var = float(stddev[0][0] ** 2) if stddev is not None else 0.0

    nuc_pixels = gray_nuc[gray_nuc > 0]
    nuc_mean = nuc_median = nuc_std = nuc_skew = nuc_kurtosis = nuc_entropy = 0.0
    glcm_contrast = glcm_corr = glcm_energy = glcm_homo = 0.0
    lbp_hist = [0.0] * 9

    if len(nuc_pixels) > 0:
        nuc_mean = float(np.mean(nuc_pixels))
        nuc_median = float(np.median(nuc_pixels))
        nuc_std = float(np.std(nuc_pixels))
        nuc_skew = float(sts.skew(nuc_pixels))
        nuc_kurtosis = float(sts.kurtosis(nuc_pixels))
        hist_vals, _ = np.histogram(nuc_pixels, bins=256, range=(0, 255), density=True)
        nuc_entropy = float(sts.entropy(hist_vals + 1e-12))

        try:
            from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
            small = cv2.resize(gray_nuc, (64, 64))
            glcm = graycomatrix(small, distances=[1],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                symmetric=True, normed=True)
            glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
            glcm_corr = float(graycoprops(glcm, 'correlation').mean())
            glcm_energy = float(graycoprops(glcm, 'energy').mean())
            glcm_homo = float(graycoprops(glcm, 'homogeneity').mean())
            del glcm, small
        except Exception:
            pass

        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray_nuc, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
            lbp_hist = lbp_hist.tolist()
            del lbp
        except Exception:
            lbp_hist = [0.0] * 9

    rg_ratio = rb_ratio = gb_ratio = 0.0
    nuc_rgb = cell_resized[nuc_mask.astype(bool)]
    if len(nuc_rgb) > 0:
        R = nuc_rgb[:, 2].astype(float).mean()
        G = nuc_rgb[:, 1].astype(float).mean()
        B = nuc_rgb[:, 0].astype(float).mean()
        rg_ratio = R / (G + 1e-6)
        rb_ratio = R / (B + 1e-6)
        gb_ratio = G / (B + 1e-6)

    return {
        'Nuc_Area': nuc_area, 'Cyt_Area': cyt_area, 'Total_Area': total_area,
        'NC_Ratio': nc_ratio, 'Nuc_Perimeter': nuc_perimeter,
        'Nuc_Circularity': nuc_circularity, 'Nuc_Eccentricity': nuc_eccentricity,
        'Nucleus_Aspect': aspect, 'Nucleus_Solidity': solidity,
        'Nucleus_Extent': extent, 'Centroid_Dist': centroid_dist,
        'Hu_1': hu[0], 'Hu_2': hu[1], 'Hu_3': hu[2], 'Hu_4': hu[3],
        'Hu_5': hu[4], 'Hu_6': hu[5], 'Hu_7': hu[6],
        'Fourier_1': fd_mag[0], 'Fourier_2': fd_mag[1], 'Fourier_3': fd_mag[2],
        'Fourier_4': fd_mag[3], 'Fourier_5': fd_mag[4],
        'Chromatin_Variance': chromatin_var,
        'Nuc_Mean': nuc_mean, 'Nuc_Median': nuc_median, 'Nuc_Std': nuc_std,
        'Nuc_Skew': nuc_skew, 'Nuc_Kurtosis': nuc_kurtosis,
        'Nucleus_Entropy': nuc_entropy,
        'GLCM_Contrast': glcm_contrast, 'GLCM_Correlation': glcm_corr,
        'GLCM_Energy': glcm_energy, 'GLCM_Homogeneity': glcm_homo,
        'LBP_0': lbp_hist[0], 'LBP_1': lbp_hist[1], 'LBP_2': lbp_hist[2],
        'LBP_3': lbp_hist[3], 'LBP_4': lbp_hist[4], 'LBP_5': lbp_hist[5],
        'LBP_6': lbp_hist[6], 'LBP_7': lbp_hist[7], 'LBP_8': lbp_hist[8],
        'RG_Ratio': rg_ratio, 'RB_Ratio': rb_ratio, 'GB_Ratio': gb_ratio,
        'Prob_0': cnn_probs[0], 'Prob_1': cnn_probs[1], 'Prob_2': cnn_probs[2],
        'Prob_3': cnn_probs[3], 'Prob_4': cnn_probs[4]
    }

# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cpu")  # CPU-only: prevents all CUDA OOM
    yolo = YOLO(r"runs\detect\sipakmed_yolo\weights\best.pt")
    
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    unet.load_state_dict(torch.load(r"trained_models\unet\unet_sipakmed.pth", map_location="cpu", weights_only=True))
    unet.eval()

    cnn = models.efficientnet_b0(weights=None)
    cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, 5)
    cnn.load_state_dict(torch.load(r"trained_models\cnn\efficientnet_sipakmed.pth", map_location="cpu", weights_only=True))
    cnn.eval()

    ensemble = joblib.load(r"trained_models\Super_Ensemble_Model.pkl")
    # Force XGBoost to run on CPU to prevent silent crashes from CUDA fallback
    try:
        ensemble.xgb.set_params(device="cpu")
        ensemble.xgb.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass
        
    scaler = joblib.load(r"trained_models\Super_Ensemble_Scaler.pkl")
    return yolo, unet, cnn, ensemble, scaler, device

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209074.png", width=60)
    st.markdown("### Clinical AI Workspace")
    patient_id = st.text_input("Patient ID", value="PT-99X")
    uploaded_file = st.file_uploader("Upload Slide Image", type=["bmp", "jpg", "png"])
    
    # ── FIX #2: Re-running works because we don't use .read() which exhausts buffer ──
    run_btn = st.button("🚀 Start Analysis", type="primary")

st.title("🔬 Clinical Diagnostics AI — Super Ensemble")

# ──────────────────────────────────────────────
# MAIN PIPELINE (Live Execution Architecture)
# ──────────────────────────────────────────────
if uploaded_file is not None:
    yolo_model, unet_model, cnn_model, ensemble_model, scaler, device = load_models()
    expected_cols = list(scaler.feature_names_in_)

    # ── FIX #2: .getvalue() fixes the bug where Re-runs cause empty image errors ──
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    scifi_lens, raw_square = apply_lens_effect(original_img)

    # ── FIX #3: Tabs are ALWAYS declared here so the UI structure is never lost ──
    tab1, tab2, tab3 = st.tabs(["Target Scanner", "Cellular Analysis", "Diagnostic Report"])

    with tab1:
        st.markdown("<div class='ar-lens-container'>", unsafe_allow_html=True)
        lens_placeholder = st.empty()
        lens_placeholder.image(resize_for_web(scifi_lens, 800))
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
                    lens_placeholder.image(resize_for_web(display_lens, 800))
                    time.sleep(0.3)

                st.success(f"Cells Detected: {len(cells_to_process)}")
                time.sleep(1)

        with tab2:
            st.markdown("### 🧬 Detailed Cellular Metrics")
            all_report_rows = []

            for idx, (x1, y1, x2, y2, cell_crop) in enumerate(cells_to_process):
                if cell_crop.size == 0:
                    continue

                focus_lens = display_lens.copy()
                cv2.rectangle(focus_lens, (x1, y1), (x2, y2), (255, 0, 100), 5)
                lens_placeholder.image(resize_for_web(focus_lens, 800))

                cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)

                unet_input = cv2.resize(cell_rgb, (128, 128)).transpose(2, 0, 1).astype('float32') / 255.0
                unet_tensor = torch.tensor(unet_input).unsqueeze(0)
                with torch.no_grad():
                    mask = torch.argmax(unet_model(unet_tensor), dim=1).squeeze().cpu().numpy()

                visual_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                visual_mask[mask == 1] = [56, 189, 248]
                visual_mask[mask == 2] = [3, 105, 161]

                gray_img = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_img, (128, 128))
                cell_mask_vis = ((mask == 1) | (mask == 2)).astype(np.uint8)
                cell_only = cv2.bitwise_and(gray_resized, gray_resized, mask=cell_mask_vis)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(cell_only)
                texture_vis = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
                texture_vis[cell_mask_vis == 0] = [0, 0, 0]

                nuc_area = float(np.sum(mask == 2))
                cyt_area = float(np.sum(mask == 1))
                nc_ratio = nuc_area / cyt_area if cyt_area > 0 else 0.0

                pil_img = Image.fromarray(cell_rgb)
                cnn_tensor = CNN_TRANSFORM(pil_img).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(cnn_model(cnn_tensor), dim=1).squeeze().numpy()

                feat_dict = extract_features_for_cell(mask, cell_crop, probs)
                feat_df = pd.DataFrame([feat_dict])[expected_cols]
                feat_scaled = scaler.transform(feat_df)
                feat_scaled_df = pd.DataFrame(feat_scaled, columns=expected_cols)

                final_class = ensemble_model.predict(feat_scaled_df)[0]
                diagnosis = CLASS_NAMES[final_class]
                ens_probs = ensemble_model.predict_proba(feat_scaled_df)[0]

                if nuc_area < 5:
                    diagnosis = "ANOMALY (NO NUCLEUS)"

                row_data = {
                    'Target': f"TGT-{idx + 1}",
                    'Diagnosis': diagnosis,
                    'Peak_Confidence': round(float(np.max(ens_probs) * 100), 2)
                }
                for k, v in feat_dict.items():
                    row_data[k] = round(v, 4) if isinstance(v, float) else v
                all_report_rows.append(row_data)

                is_danger = diagnosis in DANGER_CLASSES
                hud_class = "hud-stat-box hud-danger" if is_danger else "hud-stat-box"

                st.markdown(f"#### Cell ID: {idx + 1}")
                img_c1, img_c2, img_c3 = st.columns(3)

                with img_c1:
                    st.image(cell_rgb, caption="Raw Image", use_container_width=True)
                with img_c2:
                    st.image(visual_mask, caption="U-Net Mask", use_container_width=True)
                with img_c3:
                    st.image(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB), caption="Chromatin Texture", use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                chart_c1, chart_c2 = st.columns(2)

                with chart_c1:
                    prob_df = pd.DataFrame({'Class': CLASS_NAMES, 'Confidence (%)': ens_probs * 100})
                    fig = px.bar(prob_df, x='Confidence (%)', y='Class',
                                 orientation='h', color='Confidence (%)',
                                 color_continuous_scale='Reds' if is_danger else 'Blues')
                    fig.update_layout(
                        title="Super Ensemble Confidence", height=260,
                        margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_{idx}")
                    
                with chart_c2:
                    radar_df = pd.DataFrame({
                        'Biomarker': ['NC Ratio', 'Circularity', 'Solidity', 'Entropy', 'Texture'],
                        'Score': [
                            min(feat_dict['NC_Ratio'] * 5, 1.0), 
                            min(feat_dict['Nuc_Circularity'], 1.0),
                            min(feat_dict['Nucleus_Solidity'], 1.0),
                            min(feat_dict['Nucleus_Entropy'] / 8.0, 1.0), 
                            min(feat_dict['GLCM_Homogeneity'] * 2, 1.0) 
                        ]
                    })
                    fig_radar = px.line_polar(radar_df, r='Score', theta='Biomarker', line_close=True)
                    fig_radar.update_traces(fill='toself', line_color='#ef4444' if is_danger else '#38bdf8')
                    fig_radar.update_layout(
                        title="Cellular Biomarker Profile", height=260,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
                        margin=dict(t=30, b=20, l=40, r=40)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{idx}")

                st.markdown(f"""
                <div class='{hud_class}'>
                    <b>DIAGNOSIS:</b> {diagnosis.upper()} | <b>NC_RATIO:</b> {nc_ratio:.3f} | <b>ENSEMBLE CONFIDENCE:</b> {np.max(ens_probs)*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

                st.html(
                    "<script>window.parent.document.querySelector('.main').scrollTo("
                    "{top: window.parent.document.querySelector('.main').scrollHeight,"
                    " behavior: 'smooth'});</script>"
                )

                del unet_tensor, cnn_tensor, pil_img, feat_df, feat_scaled
                gc.collect()
                time.sleep(0.8)

        with tab3:
            st.markdown("### 📋 Final Diagnostic Report")
            if all_report_rows:
                df = pd.DataFrame(all_report_rows)
                st.markdown(f"**Patient:** {patient_id} | **Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}")
                abnormals = len(df[df['Diagnosis'].isin(DANGER_CLASSES)])

                m1, m2 = st.columns(2)
                m1.metric("TOTAL CELLS ANALYZED", len(df))

                delta_text = "CRITICAL" if abnormals > 0 else "NOMINAL"
                delta_color = "#ff4b4b" if abnormals > 0 else "#09ab3b"
                delta_bg = ("rgba(255, 75, 75, 0.15)" if abnormals > 0 else "rgba(9, 171, 59, 0.15)")

                m2.markdown(f"""
                <div style="display: flex; flex-direction: column;">
                    <span style="font-size: 14px; color: #94a3b8; font-family: sans-serif;">ANOMALIES FLAGGED</span>
                    <div style="display: flex; align-items: center; gap: 12px; margin-top: 5px;">
                        <span style="color: {delta_color}; background-color: {delta_bg}; font-size: 14px; padding: 4px 10px; border-radius: 5px; font-weight: bold;">{delta_text}</span>
                        <span style="font-size: 36px; color: white; font-weight: bold;">{abnormals}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<hr style='border-color: #334155; margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown("### 📊 Population Analytics")
                
                chart_col1, chart_col2 = st.columns(2)
                
                color_map = {
                    'Superficial-Intermediate': '#38bdf8', # light blue
                    'Parabasal': '#f59e0b',                # orange/amber
                    'Metaplastic': '#a855f7',              # purple
                    'Koilocytotic': '#ef4444',             # red
                    'Dyskeratotic': '#991b1b',             # dark red
                    'ANOMALY (NO NUCLEUS)': '#64748b'      # slate gray
                }
                
                with chart_col1:
                    class_counts = df['Diagnosis'].value_counts().reset_index()
                    class_counts.columns = ['Diagnosis', 'Count']
                    
                    fig_pie = px.pie(
                        class_counts, 
                        names='Diagnosis', 
                        values='Count',
                        hole=0.55,
                        title="Diagnosis Distribution",
                        color='Diagnosis',
                        color_discrete_map=color_map
                    )
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        margin=dict(t=50, b=20, l=0, r=0),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with chart_col2:
                    fig_scatter = px.scatter(
                        df, 
                        x='Cyt_Area', 
                        y='Nuc_Area', 
                        color='Diagnosis',
                        size='Peak_Confidence',
                        hover_data=['Target', 'NC_Ratio'],
                        title="Morphological Profiling (NC Ratio Proxy)",
                        color_discrete_map=color_map
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        margin=dict(t=50, b=20, l=0, r=0),
                        xaxis_title="Cytoplasm Area (px)",
                        yaxis_title="Nucleus Area (px)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                st.markdown("<hr style='border-color: #334155; margin: 30px 0;'>", unsafe_allow_html=True)
                st.markdown("### 🗄️ Comprehensive Feature Extraction Matrix")

                def highlight_critical(row):
                    if row['Diagnosis'] in DANGER_CLASSES:
                        return ['background-color: rgba(239, 68, 68, 0.25); font-weight: bold; color: #fca5a5'] * len(row)
                    return [''] * len(row)

                st.dataframe(df.style.apply(highlight_critical, axis=1), height=400)
