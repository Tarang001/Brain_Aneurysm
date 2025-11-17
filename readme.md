## Overview

Intracranial aneurysms affect nearly **3% of the global population**, and tragically, up to **50% are only discovered after rupture**, often leading to severe illness or death. Each year, these ruptures cause approximately **500,000 deaths worldwide**, with nearly half of the victims being **under the age of 50**.

This project was developed as part of the **RSNA Brain Aneurysm Detection Challenge**, organized in collaboration with:
- **Radiological Society of North America (RSNA)**
- **American Society of Neuroradiology (ASNR)**
- **Society of Neurointerventional Surgery (SNIS)**
- **European Society of Neuroradiology (ESNR)**

---

## Objective

The goal is to build machine learning models capable of **detecting and precisely localizing intracranial aneurysms** across different brain imaging modalities, including:

- **CTA** (Computed Tomography Angiography)
- **MRA** (Magnetic Resonance Angiography)
- **T1 post-contrast MRI**
- **T2-weighted MRI**

The dataset introduces **real clinical variability** from different institutions, scanners, and imaging protocols — challenging models to generalize effectively in diverse real-world scenarios.

---

## Why It Matters

An accurate and automated detection system can:
- **Save lives** by enabling earlier diagnosis and intervention before catastrophic rupture
- **Assist radiologists** by reducing workload and improving diagnostic efficiency
- **Standardize care** across institutions with varying imaging setups
- **Reduce healthcare costs** through early detection and prevention

---

## Results Summary

Our ensemble approach achieved significant improvements through iterative development:

| Model Version | Architecture | Strategy | AUC Score | Improvement |
|--------------|-------------|----------|-----------|-------------|
| **Model v1.0** | EfficientNetV2-S (single) | Single architecture, 1 fold | **0.50** | Baseline |
| **Model v2.0** | EfficientNetV2-S + ConvNeXt + MaxViT | 3 architectures, random fold selection | **0.58** | +16% |
| **Model v3.0** ⭐ | EfficientNetV2-S + ConvNeXt + MaxViT | 3 architectures, **best fold selection** | **0.6084** | +21.7% |

**Key Innovation in v3.0**: Instead of randomly selecting models from each fold, we automatically identify and use the **best-performing model from each fold** (3 folds × 3 architectures = 9 models), then ensemble their predictions for superior accuracy.

---

## Repository Structure
```
neuronetra-aneurysm-detection/
│
├── neuronetra-dataset-preparation.ipynb     # DICOM → PNG conversion pipeline
├── neuronetra-eda.ipynb                     # Exploratory Data Analysis
├── neuronetra_baseline_model.ipynb          # v1.0 - Single EfficientNet (AUC: 0.50)
├── neuronetra_2_0.ipynb                     # v2.0 - Random ensemble (AUC: 0.58)
├── neuronetra_3_0.ipynb                     # v3.0 - Best ensemble (AUC: 0.6084) ⭐
├── README.md                                # This file
└── requirements.txt                         # Python dependencies
```

### File Descriptions

1. **`neuronetra-dataset-preparation.ipynb`**
   - Converts DICOM medical images to PNG format
   - Applies preprocessing (windowing, CLAHE enhancement)
   - Generates compressed dataset (~80% size reduction)
   - Outputs: `cvt_png/` folder with organized PNG files

2. **`neuronetra-eda.ipynb`**
   - Exploratory Data Analysis of the dataset
   - Visualization of imaging modalities (CT, CTA, MRA, MRI)
   - Class distribution analysis (14 aneurysm locations + global presence)
   - Patient demographics and metadata statistics

3. **`neuronetra_baseline_model.ipynb`** (Model v1.0)
   - Single EfficientNetV2-Small architecture
   - Basic training setup (10 epochs)
   - Result: 0.50 AUC (baseline performance)

4. **`neuronetra_2_0.ipynb`** (Model v2.0)
   - Introduced 3-architecture ensemble
   - Random fold selection for each architecture
   - Result: 0.58 AUC (+16% improvement)

5. **`neuronetra_3_0.ipynb`** (Model v3.0) ⭐ **FINAL VERSION**
   - **Best-of-fold selection** mechanism
   - Automatically identifies highest-scoring model per fold
   - 9-model ensemble (3 architectures × 3 folds)
   - Test-Time Augmentation (horizontal flip)
   - Result: **0.6084 AUC (+21.7% improvement)**

---

## 🔬 Technical Approach

### 1️⃣ **Data Preprocessing Pipeline**

#### Why PNG Conversion?
**Challenge**: Original DICOM files are volumetric, large (~330 GB total dataset), and computationally expensive to process during training.

**Solution**: 
```
DICOM (.dcm) → Preprocessing → PNG (.png)
    ↓
 80% size reduction (330 GB → ~66 GB)
 10x faster loading during training
 Preserved medical information through proper windowing
 Enabled efficient caching and augmentation
```

**Preprocessing Steps**:
1. **Windowing**: Apply modality-specific intensity windows
   - CT: (40, 80) HU
   - CTA: (50, 350) HU
   - MRA: (600, 1200) HU
2. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
   - Enhances vessel visibility
   - Modality-specific clipLimits (2.0-3.0)
3. **Normalization**: Percentile-based robust normalization (1st-99th percentile)
4. **Resizing**: 224×224 pixels for efficient model input

---

### 2️ **Multi-Architecture Ensemble Strategy**

#### Three Complementary Architectures:

| Architecture | Backbone | Parameters | Specialty | Why Chosen |
|-------------|----------|------------|-----------|------------|
| **EfficientNetV2-Small** | `tf_efficientnetv2_s.in1k` | 21.5M | Compound scaling, balanced efficiency | Best accuracy-per-parameter ratio |
| **ConvNeXt-Tiny** | `convnext_tiny.fb_in1k` | 28M | Large kernels (7×7), local patterns | Excellent for small object detection (aneurysms: 2-7mm) |
| **MaxViT-Tiny** | `maxvit_tiny_tf_224.in1k` | 31M | Hybrid CNN + Transformer, global context | Captures vessel network topology |

**Why These Three?**
- **Architectural Diversity**: Different inductive biases lead to complementary error patterns
- **EfficientNet**: Excels at texture and fine-grained features
- **ConvNeXt**: Large receptive field captures spatial relationships
- **MaxViT**: Self-attention mechanism understands long-range vessel dependencies

When averaged, their individual weaknesses cancel out! 

---

### 3️ **3-Level Ensemble Framework**

Our final model (v3.0) uses a hierarchical ensemble strategy:
```
Level 1: Architecture Diversity (3 models)
   ├── EfficientNetV2-Small
   ├── ConvNeXt-Tiny
   └── MaxViT-Tiny

Level 2: Cross-Validation Averaging (3 folds per architecture)
   ├── Fold 0 (Best model selected automatically)
   ├── Fold 1 (Best model selected automatically)
   └── Fold 2 (Best model selected automatically)

Level 3: Test-Time Augmentation (2 predictions per model)
   ├── Original image → Prediction 1
   └── Horizontally flipped image → Prediction 2

Final Ensemble: Average of 3 × 3 × 2 = 18 predictions
```

#### **Innovation: Best-of-Fold Selection** (v3.0 improvement)

**v2.0 Approach** (Random):
```python
# Randomly picked any checkpoint from each fold
model_path = random.choice(glob.glob(f"{prefix}_fold{fold}*.pth"))
```

**v3.0 Approach** (Smart Selection):
```python
# Automatically finds the best-scoring checkpoint
model_paths = glob.glob(f"{prefix}_fold{fold}*.pth")
best_path = sorted(model_paths, 
                   key=lambda x: float(x.split('score')[-1].replace('.pth', '')))[-1]

# Example filename: effnetv2s_fold0_epoch8_score0.608421.pth
#                                              ^^^^^^^^^ Extracts this score
```

**Impact**: +0.028 AUC improvement (0.58 → 0.6084) by using only the best-trained models!

---

### 4️ **Smart 8-Frame Sampling**

Each brain scan contains 80-200 DICOM slices. We intelligently select **8 representative frames**:
```python
Strategy:
1. Skip first 10% (often poor quality/positioning slices)
2. Divide remaining frames into 8 equal segments
3. Select one frame from each segment
4. Ensures coverage of entire brain volume

Example (120 total slices):
├── Skip: Slices 0-11 (first 10%)
└── Sample: Slices [12, 25, 38, 51, 64, 77, 90, 103]
            (evenly distributed across brain)
```

---

### 5️ **3-Channel Input Encoding**

We transform 8 grayscale frames into a **3-channel RGB-like** input:
```
8 Frames (224×224 each) → 3-Channel Input (224×224×3)

Channel 1 (Red):   Middle Slice
                   ├── Representative anatomy
                   └── Frame #4 of 8 selected frames

Channel 2 (Green): Maximum Intensity Projection (MIP)
                   ├── max(all 8 frames, axis=0)
                   ├── Highlights contrast-enhanced vessels
                   └── Aneurysms appear as bright spots

Channel 3 (Blue):  Standard Deviation Projection
                   ├── std(all 8 frames, axis=0)
                   ├── Shows regions with high variation
                   └── Identifies abnormal/dynamic areas
```

**Why This Works**: 
- Pre-trained ImageNet models expect 3-channel input
- Each channel carries different spatial information
- Efficiently encodes 3D volumetric data into 2D representation

---

### 6️**Test-Time Augmentation (TTA)**

For each model prediction, we perform **horizontal flip augmentation**:
```
Original Image:          Flipped Image:
┌──────────────┐         ┌──────────────┐
│ Left   Right │  Flip   │ Right   Left │
│  ●           │  ────→  │           ●  │
│ Aneurysm     │         │     Aneurysm │
└──────────────┘         └──────────────┘

Prediction Process:
1. Model predicts on original: P1 = 0.65
2. Model predicts on flipped:  P2 = 0.67
3. Average predictions:        (P1 + P2) / 2 = 0.66

Benefits:
 Enforces left-right spatial invariance
 Reduces false negatives (catches aneurysms in both orientations)
 +0.01-0.02 AUC improvement
```

---

### 7️ **Model Training Configuration**
```python
# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 6
LEARNING_RATE = 5e-5
OPTIMIZER = AdamW(weight_decay=1e-4)
SCHEDULER = CosineAnnealingLR(T_max=10, eta_min=1e-6)

# Loss Function
ImprovedLoss(
    aneurysm_weight=3.0,    # Emphasize main detection task
    focal_weight=0.3,       # Focus on hard examples
    focal_gamma=2.0         # Down-weight easy negatives
)

# Data Augmentation
- Rotation: ±15°
- Horizontal Flip: 50%
- Brightness/Contrast: ±20%
- Gaussian Noise: σ²=10-80
- Random Gamma: 80-120%
- CoarseDropout: 8 holes (regularization)

# Cross-Validation
- GroupKFold (patient-level splitting)
- 3 folds per architecture
- Prevents data leakage from same patient
```

---

### 8️ **Inference Pipeline**

**Complete Workflow**:
```
1. Input: Test series folder with 80-200 DICOM files
   ↓
2. Smart Sampling: Select 8 representative frames
   ↓
3. Preprocessing:
   ├── DICOM reading
   ├── Windowing (modality-specific)
   ├── CLAHE enhancement
   └── Resize to 224×224
   ↓
4. 3-Channel Encoding:
   ├── Middle slice
   ├── MIP
   └── Std projection
   ↓
5. Normalization:
   ├── TensorFlow-style: mean=[0.5, 0.5, 0.5] (EfficientNetV2)
   └── PyTorch-style: ImageNet stats (ConvNeXt, MaxViT)
   ↓
6. Ensemble Inference:
   ├── 9 models × 2 TTA = 18 predictions
   └── Average all predictions
   ↓
7. Output: 14 probabilities
   ├── 13 anatomical locations
   └── 1 global "Aneurysm Present"
   ↓
8. Submission: Polars DataFrame → Competition server

Time per series: ~4 seconds
```

---

##  Challenges Faced & Solutions

### 1️ **Massive Dataset Size (330 GB)**
**Problem**: 
- Original DICOM dataset: 330 GB
- Kaggle storage limits: 100 GB per dataset
- Slow I/O during training

**Solution**:
-  Converted DICOM → PNG (80% size reduction)
-  Compressed dataset: ~66 GB
-  Enabled efficient caching during training
-  10x faster data loading

---

### 2️ **Kaggle Time Limits (12-hour maximum)**
**Problem**:
- Training 9 models (3 architectures × 3 folds)
- Each model needs 10 epochs
- 12-hour hard limit on Kaggle notebooks

**Solution**:
-  PNG conversion allowed faster training
-  Trained each architecture separately
-  Used gradient accumulation (effective batch size = 30)
-  Mixed precision training (FP16) for 2x speedup
-  Persistent workers to reduce data loading overhead

**Training Time Breakdown**:
```
Per Model (1 architecture, 1 fold):
├── Data loading: ~1.5 hours
├── Training (10 epochs): ~4 hours
└── Validation: ~0.5 hours
Total: ~6 hours per model

All 9 Models:
├── Sequential training: ~54 hours (impossible!)
├── Solution: Trained separately across multiple sessions
└── Used saved checkpoints for inference ensemble
```

---

### 3️⃣ **Limited Computational Resources**
**Problem**:
- Kaggle GPU: Tesla T4 (16 GB VRAM)
- Large models + ensemble = memory constraints
- Needed to fit 9 models for inference

**Solution**:
-  Selected efficient architectures (21-31M parameters)
-  Used mixed precision inference (50% memory reduction)
-  Batch size optimization (6 samples)
-  Gradient accumulation (5 steps → effective batch = 30)
-  Memory cleanup between folds (`gc.collect()`, `torch.cuda.empty_cache()`)

**Memory Usage**:
```
Training (single model):
├── Model weights: ~300 MB
├── Optimizer states: ~600 MB
├── Batch + activations: ~1.5 GB
├── Cached data: ~500 MB
└── Total: ~3 GB / 16 GB (19% utilization) 

Inference (9 models):
├── 9 model weights: ~2.5 GB
├── Single batch: ~200 MB
├── Activations: ~300 MB
└── Total: ~3 GB / 16 GB (19% utilization) 
```

---

### 4️ **Class Imbalance**
**Problem**:
- Aneurysms are rare (~10-15% positive rate)
- 14 different locations with varying frequencies
- Standard loss functions fail

**Solution**:
-  Focal Loss (focuses on hard examples)
-  Weighted BCE (3x weight on "Aneurysm Present")
-  Combined loss: 70% weighted BCE + 30% focal
-  Data augmentation to increase positive samples

---

### 5️ **Multi-Modality Imaging**
**Problem**:
- 4 different scan types (CT, CTA, MRA, MRI)
- Different intensity ranges and characteristics
- Models need to generalize across modalities

**Solution**:
-  Modality-specific windowing
-  CLAHE with modality-adaptive parameters
-  Robust percentile-based normalization
-  Large diverse training set

---

##  Evaluation Metrics

**Competition Metric**: Average of two AUC scores
```
Final Score = (AUC_locations + AUC_aneurysm_present) / 2

Where:
├── AUC_locations: Mean AUC across 13 anatomical locations
└── AUC_aneurysm_present: Binary classification AUC
```

**Target Locations** (13 + 1 global):
1. Left Infraclinoid Internal Carotid Artery
2. Right Infraclinoid Internal Carotid Artery
3. Left Supraclinoid Internal Carotid Artery
4. Right Supraclinoid Internal Carotid Artery
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation
14. **Aneurysm Present** (Global Detection)

---

### Requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
opencv-python>=4.8.0
pydicom>=2.4.0
pandas>=2.0.0
polars>=0.19.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

##  Usage

### 1. Dataset Preparation
```bash
# Run DICOM to PNG conversion
jupyter notebook neuronetra-dataset-preparation.ipynb

# This will:
# - Read DICOM files from input directory
# - Apply windowing and CLAHE
# - Save as PNG files (80% size reduction)
# - Generate series_index_mapping.csv
```

### 2. Exploratory Data Analysis
```bash
# Analyze dataset characteristics
jupyter notebook neuronetra-eda.ipynb

# Visualizations include:
# - Class distribution
# - Imaging modality breakdown
# - Patient demographics
# - Sample images per location
```

### 3. Training
```bash
# Train baseline model (v1.0)
jupyter notebook neuronetra_baseline_model.ipynb

# Train improved ensemble (v2.0)
jupyter notebook neuronetra_2_0.ipynb

# Train final best ensemble (v3.0) ⭐
jupyter notebook neuronetra_3_0.ipynb
```

### 4. Inference

Model v3.0 includes automatic inference via `rsna_inference_server`:
```python
# Load 9 best models automatically
loaded_models = load_all_models()

# Predict on test series
predictions = predict(series_path="/path/to/dicom/folder")

# Returns: Polars DataFrame with 14 probabilities
```

---

##  Model Evolution & Learnings

### Model v1.0 → v2.0 (+16% improvement)
**Key Changes**:
- Single architecture → 3 architectures
- No ensemble → 9-model ensemble
- Basic augmentation → Strong augmentation

**Lesson**: Architectural diversity is crucial for medical imaging

---

### Model v2.0 → v3.0 (+5% improvement)
**Key Changes**:
- Random fold selection → **Best-of-fold selection**
- No TTA → Horizontal flip TTA
- Standard normalization → Architecture-specific normalization

**Lesson**: Careful model selection and proper preprocessing matter more than adding complexity

---

##  Performance Analysis

### Individual Architecture Performance
```
EfficientNetV2-Small:
├── Fold 0: 0.586 AUC
├── Fold 1: 0.582 AUC
├── Fold 2: 0.590 AUC ← Best (selected in v3.0)
└── Average: 0.586 AUC

ConvNeXt-Tiny:
├── Fold 0: 0.575 AUC
├── Fold 1: 0.585 AUC ← Best
├── Fold 2: 0.580 AUC
└── Average: 0.580 AUC

MaxViT-Tiny:
├── Fold 0: 0.592 AUC ← Best
├── Fold 1: 0.588 AUC
├── Fold 2: 0.590 AUC
└── Average: 0.590 AUC (Best individual architecture)

Final Ensemble (v3.0): 0.6084 AUC ⭐
```

**Observation**: Ensemble outperforms even the best individual model by +0.018 AUC!

---

##  Future Improvements

### Short-term (No Retraining Required)
- [ ] Weighted ensemble based on validation scores
- [ ] Additional TTA variants (rotation, scaling)
- [ ] Post-processing with temperature scaling
- [ ] Threshold optimization per anatomical location

### Medium-term (Retraining Needed)
- [ ] Increase training epochs (10 → 20-30)
- [ ] Larger image size (224 → 256 or 384)
- [ ] Add 4th architecture (Swin Transformer)
- [ ] 5-fold cross-validation (more robust)

### Long-term (Architecture Changes)
- [ ] 3D CNN models (process full volumes)
- [ ] Attention-based models with spatial localization
- [ ] Multi-task learning (detection + segmentation)
- [ ] Self-supervised pre-training on medical images

---

##  Acknowledgments

- **RSNA, ASNR, SNIS, ESNR** for organizing the competition
- **Kaggle** for providing computational resources
- **PyTorch Image Models (timm)** library by Ross Wightman
- Medical imaging community for open-source tools and research
