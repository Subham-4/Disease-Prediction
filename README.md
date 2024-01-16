# Disease Prediction System

## Overview

This project is a Medical Diagnosis System that utilizes machine learning models to predict the likelihood of various health conditions based on user inputs or medical scans. The system covers a range of health issues, including stroke, heart disease, liver disease, kidney disease, diabetes, and Alzheimer's disease. Additionally, it provides a symptom-based diagnosis for common ailments.

## Features

1. **Stroke Prediction:**
   - Utilizes a machine learning model to predict the likelihood of a stroke based on user-provided data.

2. **Heart Disease Prediction:**
   - Predicts the probability of heart disease using a machine learning model trained on relevant features.

3. **Liver Disease Prediction:**
   - Utilizes a machine learning model to assess the likelihood of liver disease based on user-input data.

4. **Kidney Disease Prediction:**
   - Predicts the probability of kidney disease using a machine learning model trained on relevant features.

5. **Diabetes Prediction:**
   - Predicts the likelihood of diabetes based on user-provided data using a trained machine learning model.

6. **Alzheimer's Disease Detection:**
   - Analyzes brain scans to detect Alzheimer's disease and categorizes its severity.

7. **Symptom-based Disease Prediction:**
   - Allows users to input their symptoms and predicts possible diseases based on a pre-trained symptom-based machine learning model.

## Technologies Used

- Flask: Web framework for creating a user-friendly interface.
- TensorFlow: Deep learning library for training and deploying machine learning models.
- OpenCV: Image processing library for handling medical scans.
- HTML/CSS: Front-end design for web pages.
- Python: Backend language for server-side logic.

## Model Details

- The project uses machine learning models for each health condition, trained on relevant datasets.
- Models are stored in JSON format and loaded using the `model_from_json` function from TensorFlow's Keras.
- Symptom-based prediction uses a pre-trained model with symptoms as features and diseases as labels.

## Output
- The system provides predictions for various health conditions based on the input data or medical scans with more than 97% accuracy.
- Users receive a percentage likelihood of having a particular health condition.
- The system categorizes Alzheimer's disease severity based on brain scans.
- Symptom-based predictions output the most likely diseases based on user-provided symptoms.

![home](https://github.com/Subham-4/Disease-Prediction/assets/84079854/4177e300-0947-42f0-bd6a-2fb6e6722722)
![heart](https://github.com/Subham-4/Disease-Prediction/assets/84079854/ed13acf8-8091-42dd-8ea5-28610ccb77b9)
![symptoms](https://github.com/Subham-4/Disease-Prediction/assets/84079854/b48c12a0-e934-4280-99e2-a7565c95e0ba)
![alzheimer](https://github.com/Subham-4/Disease-Prediction/assets/84079854/3b31d3e8-18cd-43cb-bd03-c2ae19656237)



## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/medical-diagnosis-system.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

   The application will be accessible at `http://localhost:5000` in your web browser.

## Acknowledgments

- The machine learning models used in this project are trained on publicly available datasets. Credits to the respective dataset creators and contributors.


## Demo Video Link

- https://youtu.be/2Kf84-EwHYU
