# Nationality-Deduction
# Face Attribute AI

## Project Description

Face Attribute AI is a computer vision application that analyzes
an uploaded facial image and provides:

- Demographic group
- Estimated age
- Facial emotion
- Approximate dress colour

The application provides a graphical interface using Streamlit.

## Technologies

- Python
- DeepFace
- TensorFlow
- OpenCV
- NumPy
- Pillow
- Streamlit

## Dataset

LNS Human Face Dataset.

The dataset contains facial images with variation in
position, skin colour, nationality, age groups and expressions.

## Important Limitation

The model's demographic prediction should not be interpreted
as verified nationality. Nationality cannot reliably be
determined from facial appearance alone.

## Installation

Create virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## Application Workflow

1. Upload image
2. Preview image
3. Detect face
4. Predict demographic group
5. Predict age
6. Predict emotion
7. Estimate dress colour
8. Display conditional results
