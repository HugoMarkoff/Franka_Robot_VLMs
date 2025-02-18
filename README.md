# Franka_Robot_VLMs

## Welcome! 👋
Hello and welcome to this repository!  

This repository is dedicated to testing **state-of-the-art Vision-Language Models (VLMs)** for **industrial applications** involving the **Franka Emika Robot**. Our goal is to evaluate how well modern VLMs can perform in real-world industrial robotics scenarios.

---

## Overview
This project allows for benchmarking different **Vision-Language Models (VLMs)** in an industrial setting. It includes:
1. **Loading and running various SOTA (State-of-the-Art) VLMs** such as BLIP, CLIP, Llava, and others.
2. **Performing object detection and attribute recognition** with images from an industrial environment.
3. **Testing spatial relationships and reasoning tasks** to assess the practical use of VLMs in robotic manipulation.

You are free to use this code, but be aware:
- Some models (e.g., BLIP, CLIP, Llava) may have **licenses that restrict commercial use**.  
  Please check their respective licenses before deployment in production settings.

---

## Getting Started

### Step 1: Create a Virtual Environment
To keep dependencies organized, it's recommended to create a virtual environment.  

- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate

### Step 2: Install Dependencies
Once inside the virtual environment, install all required dependencies:
pip install -r requirements.txt
## install PyTorch with CUDA 11.8 support manually:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 -f https://download.pytorch.org/whl/cu118/torch_stable.html


### Hardware Requirements
GPU: Minimum 10GB VRAM (RTX 3080, 3090, 4070, or higher recommended).
Storage: At least 30GB of free space for downloading models and datasets.
CUDA: Ensure that NVIDIA CUDA drivers are properly installed for GPU acceleration.