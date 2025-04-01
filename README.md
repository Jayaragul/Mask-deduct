# Face Mask Detection 🚀

This project uses **MobileNetV2** to classify whether a person is wearing a mask or not. The model is trained on the **Face Mask 12K Images Dataset** and deployed using **Gradio** for easy accessibility.

## 📂 Dataset
The dataset contains images of:
- **With Mask**  
- **Without Mask**  
It is split into **Train, Validation, and Test** sets.

## 🏗 Model Training
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Data Augmentation**: Rescaling, Rotation, Zoom, Flip
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Epochs**: 10

## 🚀 Deployment
We use **Gradio** to build a simple web interface.  
To run locally:
```bash
pip install -r requirements.txt
python app.py

i have also deployed in hugging face
