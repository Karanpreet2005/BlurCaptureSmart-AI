# 📸 CapSmartAI

CapSmartAI is a cutting-edge mobile photography assistant that leverages artificial intelligence to dramatically enhance image quality. Unlike standard auto-mode features, CapSmartAI analyzes scenes in real-time using deep neural networks to recommend optimal camera settings, detect image issues, and provide instant visual feedback — all without requiring deep photography knowledge.

---

## 🌟 Key Features

### 🔧 Intelligent Camera Setting Recommendations
- **Dynamic ISO Optimization**: Balances lighting and noise for ideal ISO selection.
- **Shutter Speed Intelligence**: Recommends shutter speed based on motion and light.

### 🔍 Advanced Blur Detection and Analysis
- **Multi-Scale Blur Quantification**: Measures blur severity on a 0–100 scale
- **Region-Specific Analysis**: Identifies and maps blur-heavy zones using heatmaps.
- **Before/After Comparison**: Real time comparison analysis. Visualizes improvements when recommended settings are applied

### 🖼️ Visual Feedback Systems
- **Motion Blur Heatmaps**: Color-coded overlays indicate blur zones.
- **Real-Time Metrics Display**: Quality scores shown live during composition.

### 🎮 User Experience
- **Dual Control Modes**: Switch between AI and manual controls.
---

## 🧠 Technical Architecture

### 🤖 Deep Learning Pipeline

#### Blur Detection System
- **Model**: Modified MobileNetV2
- **Input**: 224×224 RGB images, normalized [0,1]
  
- GlobalAveragePooling2D
Dense(128, ReLU) + Dropout(0.3)
Dense(64, ReLU) + Dropout(0.2)
Dense(1, linear) → blur severity score

- **Training**: 16,000+ synthetic images
- **Augmentations**: Global average pooling, gaussian blur, motion blur
- Ratio of Motion blurred images to Gaussian is 2:1

#### ISO Prediction
- **Architecture**: MLP classifier
- **Features**: Brightness, histogram stats, edge density, noise
- **Training Strategy**: Downweight rare high-ISO cases

#### Shutter Speed Prediction
- **Architecture**: Regression on log(1/shutter)
- **Inputs**: Motion, edges, subject distance, light
- **Safety**: Constraints to avoid extreme values
- **Exposure Compensation**: Integrated with ISO logic

---

## 🌐 API Server

The system is powered by a Flask-based REST API with the following components:

Core Endpoints
/recommend_settings:

Method: POST
Input: JPEG image (multipart/form-data)
Processing:
Image normalization and feature extraction
ISO prediction via MLP classifier
Shutter speed prediction via regression model
Confidence-based adjustment with exposure preservation
Output: JSON with recommended settings and confidence metrics
{
  "iso": 400,
  "shutter_speed": 0.008,
  "iso_confidence": 0.92
}

#### `/generate_heatmap`

Method: POST
Input: JPEG image (multipart/form-data)
Processing:
Edge detection and gradient analysis
Motion vector estimation
Blur probability mapping
Color-coded visualization generation
Output: JPEG image with color overlay highlighting blur regions

### Server Configuration
- Flask-based REST API
Local hosting during demos is smart because it minimizes delay, avoids networking issues, and keeps your setup simple and self-contained.

---

## 📱 Android Application

### Core Components
MainActivity: Camera preview and primary controls
ComparisonActivity: Side-by-side image comparison
HeatmapActivity: Blur visualization interface
SettingsActivity: User preference management
BaseDrawerActivity: Navigation drawer implementation

### Camera System
Built on CameraX API for modern Android devices
Camera2 interop for advanced manual controls
Custom image analyzer for real-time frame processing

### UI Features
- Material Design 3
- Camera overlay and feedback views
- Touch controls and animations
  

---

## 📊 Performance Metrics

| Metric                    | Value                                 |
|---------------------------|---------------------------------------|
| Blur Detection Accuracy   | 94% correlation w/ human ratings      |
| ISO Recommendation        | 87% match to expert choices           |
| Shutter Speed Accuracy    | 92% within 1-stop of optimal          |
| On-Device Analysis Time   | < 100ms per frame                     |
| API Round-Trip Time       | < 80-90ms(For locally hosted server)  |

---

## 🛠️ Implementation Details

### Training Data
Synthetic Dataset: Programmatically generated blur variations applied to clear images
Real-world Dataset: 5,000+ manually captured photos with associated metadata
Expert-labeled Images: Professional assessment of optimal settings for reference scenes

### Key Files
- `feature_extraction.py`: Image analysis logic
- `settings_api.py`: Flask API endpoints
- `Constants.kt`: Android server config
- `MainActivity.kt`: Main camera view

---

The blur detection combines multiple approaches:

def composite_blur_score(image):
    # Edge-based measurement
    laplacian_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    # Gradient-based measurement
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    tenengrad_score = np.mean(gx**2 + gy**2)
    
    # Perceptual blur measurement (edge width analysis)
    pbm = perceptual_blur_metric(gray_image)
    
    # Deep learning model prediction
    dl_score = blur_model.predict(preprocess(image))[0][0]
    
    # Weighted composite
    return 0.2*normalize(laplacian_score) + 0.2*normalize(tenengrad_score) + 
           0.1*normalize(pbm) + 0.5*dl_score

      

## 🚀 Getting Started

### Prerequisites

**Server:**
- Python 3.8+
- TensorFlow 2.5+
- OpenCV 4.5+
- Flask 2.0+
- NumPy, SciPy, scikit-learn

**Android:**
- Android Studio Arctic Fox+
- Device with Camera2 API (Android 7.0+)
- 
### Server Setup
```bash
git clone https://github.com/Karanpreet2005/BlurCaptureSmart-AI.git
cd capsmartai
pip install -r requirements.txt
python settings_api.py

Project Structure:

CapSmartAI/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── settings_api.py               # Main API server
├── blur_detection_train.ipynb    # Blur detection model training
├── iso_train.py                  # ISO prediction model training
├── ss_train.py                   # Shutter speed model training
├── models/                       # Pre-trained model files
│   ├── blur_detection_model_v2.h5
│   ├── iso_classifier_model.h5
│   ├── stable_shutter_model.h5
│   └── ...
├── api/                          # API implementation modules
│   ├── blur_api.py               # Blur detection API functions
│   ├── feature_extraction.py     # Image feature extraction
│   └── ...
├── utils/                        # Utility functions
│   ├── image_processing.py
│   ├── visualization.py
│   └── ...
└── app/                          # Android application
    ├── src/main/                 # App source code
    │   ├── java/com/example/dummy/
    │   │   ├── MainActivity.kt
    │   │   ├── HeatmapActivity.kt
    │   │   ├── ComparisonActivity.kt
    │   │   ├── UnblurActivity.kt
    │   │   └── ...
    │   └── res/                  # Android resources
    └── ...



🔮 Future Roadmap
Advanced Subject Detection: Identify main subjects for more targeted recommendations
Artistic Style Transfer: Suggest settings based on desired photo style
Multi-frame Capture: HDR and noise reduction through multiple exposure blending
Real-time Video Analysis: Extend capabilities to video capture
Offline Model Inference: Full on-device processing without server dependency






