# Diabetes Prediction Using Deep Learning on Pima Indian Diabetes Dataset

## Project Overview
This project implements a deep learning solution using a Convolutional Neural Network (CNN) to predict diabetes based on the Pima Indian Diabetes dataset. The model is trained to accurately classify individuals as diabetic or non-diabetic based on various health measurements.

## Key Features
- Utilized a CNN architecture tailored for structured medical data classification.
- Trained the model for 200 epochs with a batch size of 8.
- Achieved a test accuracy of 81%, demonstrating effective prediction performance.
- Employed robust data preprocessing and model evaluation techniques.

## Technologies Used
- Programming Language: Python
- Deep Learning Framework: TensorFlow / Keras
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib / Seaborn
- Dataset: Pima Indian Diabetes Dataset

## Project Structure
- `data/` — Contains the Pima Indian Diabetes dataset CSV file.
- `models/` — Saved CNN model files and checkpoints.
- `notebooks/` — Jupyter notebooks for data exploration, model building, training, and evaluation.
- `scripts/` — Python scripts for training and testing the CNN model.
- `README.md` — This documentation file.
- `requirements.txt` — Required Python packages and versions.

## Usage
1. Clone the repository.
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the training script or Jupyter notebook to train the CNN model.  
4. Evaluate the trained model on test data to verify accuracy.

## Results
- The CNN model achieved **81% accuracy** on the test set after training for 200 epochs with batch size 8.
- The model demonstrates promising capability for diabetes prediction based on clinical attributes.

## Future Work
- Experiment with advanced architectures and hyperparameter tuning to improve accuracy.
- Incorporate additional feature engineering and data augmentation methods.
- Deploy the model as an API or integrate with healthcare applications.


