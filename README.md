# SIPaKMeD Cervical Cancer Detection: Deep Learning Ensemble Pipeline

## Project Architecture Overview
This project implements an advanced, multi-stage Deep Learning pipeline designed to detect and classify cervical cancer cells from the SIPaKMeD dataset. Rather than relying on a single model, this architecture uses an **Ensemble Meta-Learner approach**. We deploy specialized Neural Networks for distinct tasks (Detection, Measurement, and Texture Analysis) and mathematically fuse their outputs into a final XGBoost classifier to achieve maximal accuracy.

---

## Directory Structure
To keep the project organized and prevent data leakage, the repository is split into specific directories based on their role in the pipeline:

*   **`Dataset/`**: The original, raw SIPaKMeD dataset containing uncropped microscopic images (`.bmp`) and their corresponding cell coordinate annotations (`.dat`).
*   **`notebooks/`**: Contains the 8 sequential Jupyter Notebooks that drive the entire Deep Learning pipeline from data preprocessing to final deployment.
*   **`processed_data/`**: Contains the extracted, cropped datasets formatted specifically for YOLO (bounding boxes), U-Net (categorical masks), and EfficientNet (class-sorted folders). It also contains the final `extracted_features.csv` file.
*   **`trained_models/`**: The vault for our trained AI weights. This prevents us from having to retrain models from scratch. It contains `unet_sipakmed.pth`, `efficientnet_sipakmed.pth`, and `xgboost_sipakmed.json`.
*   **`runs/`**: A system-generated folder created by the Ultralytics YOLOv8 library. It stores training logs, loss graphs, and visual validation predictions from the Object Detection phase.
*   **`requirements.txt`**: A standard Python dependency file containing all required libraries needed to run this project.
*   **`sipakmed_env/`**: The isolated Python Virtual Environment that safely contains all the installed packages for this project.

---

## Data Flow & Notebook Architecture

### Phase 1: Data Preparation
**File:** `notebooks/1_Data_Preprocessing.ipynb`
*   **Exact Purpose:** To translate the raw, unstructured medical dataset into highly specific formats required by our three different Deep Learning architectures.
*   **Data Input:** Raw `.bmp` images and `.dat` coordinate files provided by the original SIPaKMeD pathologists.
*   **Data Output:** YOLO Dataset, U-Net Dataset, and CNN Dataset.

### Phase 2: Specialized Model Pre-Training
In this phase, we train three separate Artificial Intelligences to perform distinct, highly specialized medical tasks.

**File:** `notebooks/2_YOLOv8_Training.ipynb`
*   **Exact Purpose:** To act as the "Cell Hunter." It learns to look at a massive, messy microscope slide and draw physical bounding boxes around the cells of interest so they can be isolated.
*   **Data Output:** `yolov8n_sipakmed.pt`

**File:** `notebooks/3_UNet_Training.ipynb`
*   **Exact Purpose:** To act as the "Morphological Measurement AI." We use Transfer Learning (ResNet-34) to train a U-Net semantic segmentation model. Its job is to physically paint over the cell at the pixel level to isolate the Nucleus and the Cytoplasm.
*   **Data Output:** `unet_sipakmed.pth`

**File:** `notebooks/4_EfficientNet_Training.ipynb`
*   **Exact Purpose:** To act as the "Texture Classification AI." We use a pre-trained EfficientNet-B0 Convolutional Neural Network (CNN). It ignores physical size and purely analyzes the microscopic textures, chromatin patterns, and color densities of the cell to guess its diagnosis.
*   **Data Output:** `efficientnet_sipakmed.pth`

### Phase 3: Data Fusion (Feature Extraction)
**File:** `notebooks/5_Feature_Extraction.ipynb`
*   **Exact Purpose:** To bridge the gap between Deep Learning (images) and Machine Learning (math). We lock the U-Net and EfficientNet models in "evaluation mode" and pass the entire dataset through them to extract raw numbers.
*   **Data Flow:**
    1.  **U-Net Pipeline:** Image -> U-Net -> Pixel Mask -> Mathematical calculation of the Nucleus-to-Cytoplasm (NC) Ratio.
    2.  **EfficientNet Pipeline:** Image -> EfficientNet -> Softmax Layer -> Exact percentage probabilities for all 5 medical classes.
*   **Data Output:** `extracted_features.csv`

### Phase 4: The Meta-Learner
**File:** `notebooks/6_XGBoost_and_Evaluation.ipynb`
*   **Exact Purpose:** To act as the "Chief Medical Officer." It takes the mathematical measurements from U-Net and the texture probabilities from EfficientNet and learns the absolute best way to combine them.
*   **Data Output:** `xgboost_sipakmed.json`, Classification Report, and Confusion Matrix Heatmap.

### Phase 5: Scientific Explainability & Ablation
**File:** `notebooks/7_Ablation_and_Explainability.ipynb`
*   **Exact Purpose:** To mathematically and visually prove that our complex architecture is justified and to prevent "black box" AI concerns.
*   **Data Flow:**
    1.  **Ablation Study:** A mathematical breakdown proving that the Full Ensemble (U-Net + CNN + XGB) vastly outperforms any of the individual models operating alone.
    2.  **Grad-CAM Explainability:** Generates a thermal heatmap over the cell images showing exactly which physical pixels the EfficientNet AI was looking at when it made its medical diagnosis.

### Phase 6: Final Deployment Pipeline
**File:** `notebooks/8_Final_Pipeline.ipynb`
*   **Exact Purpose:** The ultimate deployment script. It integrates all 4 AI models into a single, automated clinical pipeline capable of diagnosing a completely raw microscope slide.
*   **Data Flow:** Raw Image -> YOLOv8 (Detection & Cropping) -> U-Net (Morphology) -> EfficientNet (Texture) -> XGBoost (Final Diagnosis) -> Drawn bounding box with text overlay.
