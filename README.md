# Retinal OCT Disease Classification using MobileViT-XS

This project implements a lightweight MobileViT-XS model to classify retinal OCT (Optical Coherence Tomography) images into four categories:

- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

The model is trained on a 10% stratified subset of the OCT2017 dataset and achieves high accuracy while remaining efficient enough for CPU inference and deployment.

---

## Reference Paper

This project is based on the architecture described in:

MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer  
Sachin Mehta, Mohammad Rastegari  
Paper Link: https://arxiv.org/abs/2110.02178

Key ideas used from the paper:
- Combination of CNNs (local features) and Transformers (global features)
- Patch-based global attention
- Lightweight architecture suitable for mobile/edge deployment

---

## Project Structure

oct_mobilevit_app/  
├── mobilevit_model.py          # MobileViT-XS architecture  
├── mobilevit_xs_best.pth       # Trained model weights  
├── infer.py                    # Single-image inference script  
├── app.py                      # Streamlit web app  
├── requirements.txt            # Dependencies  
└── README.md  

---

## Installation

### 1. Clone the Repository

git clone https://github.com/Mukheshkumar25/Transformer-based-OCT-images-classification.git  
cd Transformer-based-OCT-images-classification

### 2. Create & Activate Virtual Environment

Windows:
python -m venv .venv  
.venv\Scripts\activate  

macOS/Linux:
python3 -m venv .venv  
source .venv/bin/activate  

### 3. Install Dependencies

pip install -r requirements.txt

---

## Run Inference on a Single Image

Place an OCT image (JPG/PNG) inside the project folder.

Run:

python infer.py

Example output:
Prediction: CNV (96.12% confidence)

---

## Run Streamlit Web Application

To launch the web interface:

streamlit run app.py

This will open a local browser at:
http://localhost:8501

Upload any OCT image and the app will classify it using the trained MobileViT-XS model.

---

## Model Performance (10% OCT2017 Subset)

Accuracy: 0.9407  
Precision: 0.9090  
Recall: 0.9281  
F1 Score: 0.9179  

---

## Dataset

This project uses the publicly available OCT2017 Retina Dataset which contains four classes:

- CNV  
- DME  
- DRUSEN  
- NORMAL  

A 10% stratified subset was used to reduce compute load while maintaining class representation.

Dataset link (Kaggle): https://www.kaggle.com/paultimothymooney/kermany2018

---

## Disclaimer

This model is intended solely for research and educational purposes.  
It is not approved for clinical diagnosis or medical decision-making.

---

## Acknowledgements

- MobileViT paper authors  
- OCT2017 dataset creators  
- PyTorch & Streamlit communities  
