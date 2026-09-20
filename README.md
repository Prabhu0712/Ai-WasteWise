# ♻️ AI WasteWise

AI-powered waste classification and sustainable waste-management assistant.

## Features

- 30-class waste classification
- EfficientNetV2B0 deep learning model
- 84.10% test accuracy
- Grad-CAM explainability
- RAG-based waste-management knowledge
- Disposal recommendations
- Reuse recommendations
- Sustainability tips
- Streamlit web application

## Model

The system uses a fine-tuned EfficientNetV2B0 model trained on a 30-class household and recyclable waste dataset.

## Pipeline

Image
→ EfficientNetV2B0
→ Waste Classification
→ Confidence Score
→ Grad-CAM
→ RAG Knowledge Retrieval
→ Disposal & Sustainability Guidance

## Technology

- Python
- TensorFlow / Keras
- EfficientNetV2B0
- Streamlit
- NumPy
- Matplotlib
- Grad-CAM
- RAG

## Deployment

The application is deployed using Streamlit Community Cloud.
