# Tag-init: Physics-Informed Thermal Super-Resolution (ML Core)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![EarthEngine](https://img.shields.io/badge/Google_Earth_Engine-Cloud_Computing-4285F4.svg)](https://earthengine.google.com/)

This repository contains the core Machine Learning research and training pipeline for **Project Tag-init**. The project focuses on **Thermal Super-Resolution**: transforming 100m/30m resolution thermal data (Landsat) into 10m high-resolution heat maps using Sentinel-2 RGB guidance and Physics-Informed Neural Networks (PINNs).

---

## Purpose & Impact

The goal of **Tag-init** is to solve the "spatial blindness" of current urban heat monitoring. While satellite thermal sensors are restricted to coarse 100m resolutions, this project uses **Thermal Super-Resolution** to predict heat at a 10m granular scale.

### What it does:
* **Visualizes the Invisible:** Transforms standard RGB satellite imagery into high-fidelity thermal maps.
* **Material Intelligence:** Learns the specific heat-retention patterns of Philippine urban materials (e.g., G.I. roofs vs. asphalt).
* **Bridging Research to Reality:** Provides the mathematical foundation for a real-time mobile AR tool, turning raw environmental data into actionable street-level insights for urban cooling.

---

## Technical Evolution: Beyond the MLP
This project represents a significant architectural shift from my previous work in Urban Heat Island (UHI) prediction:

| Feature | Project v1 (MLP) | **Project v2 (Tag-init PINN)** |
| :--- | :--- | :--- |
| **Logic** | Point-wise Regression | **Spatial Feature Extraction** |
| **Input** | Tabular and Generated Data | **256x256 Image Patches** |
| **Architecture** | Simple Neural Net (MLP) | **U-Net (Encoder-Decoder)** |
| **Physics** | Statistical Constraints | **Laplacian Diffusion Residuals** |

---

## Scientific Methodology

### 1. Data Engineering (Tag-init Focus)
To ensure the model is robust for the Philippine context, we focus exclusively on the **"Tag-init"** months (March – May). 
* **Satellite Fusion:** Sentinel-2 (10m RGB) fused with Landsat 8 (resampled 10m Thermal).
* **Temporal Filtering:** Multi-year median compositing (2021–2025) to eliminate cloud artifacts.
* **Spatial Slicing:** Sliding-window generation (Stride: 64) resulting in ~1,000 unique training patches of Metro Manila.

### 2. PINN Architecture: The U-Net
The model uses a U-Net backbone to maintain high-resolution spatial information. Skip connections are utilized to pass fine-grained urban textures (like building edges and road narrowness) directly to the decoding layers.

### 3. Physics-Informed Loss Function
Unlike standard CNNs that only use Mean Squared Error (MSE), this model incorporates the **Heat Diffusion Equation** as a regularization term:
$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{physics}$$
The physics loss uses a **Laplacian Kernel** ($\nabla^2$) to penalize physically impossible heat spikes and ensure thermal continuity across material boundaries.

---
## Mobile Deployment

The final model is optimized using 8-bit Static Quantization (QNNPACK) for real-time inference.

- Original Size: ~120MB

- Quantized Size: ~30MB

- Target Platform: Android (Companion Repo: To be made soon)

## Setup & Usage
1. Open `taginit_pipeline.ipynb` in Google Colab.
2. Authenticate Google Earth Engine.
3. Run the Data Slicer to generate patches.
4. Execute the training loop with PhysicsInformedLoss
