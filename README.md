# Stress Detection using Machine Learning

Machine Learning project focused on detecting stress and physical effort levels using physiological wearable sensor data.

## Overview

This project classifies physiological states into:

- Stress
- Aerobic Activity
- Anaerobic Activity

The models were trained using physiological signals collected from wearable sensors, including:

- Heart Rate (HR)
- Electrodermal Activity (EDA)
- Skin Temperature (TEMP)
- Accelerometer (ACC)
- Blood Volume Pulse (BVP)
- Inter-Beat Interval (IBI)

## Dataset

The project uses physiological signal datasets inspired by WESAD and wearable stress detection research.

## Feature Engineering

Several statistical and physiological features were extracted:

- Mean
- Standard deviation
- Min/Max
- Quartiles
- Accelerometer magnitude
- HRV (RMSSD)

## Models Tested

The following models were evaluated:

- Logistic Regression
- k-NN
- Naive Bayes
- SVM
- MLP Neural Network
- XGBoost

## Best Results

| Model | AUC |
|------|------|
| MLP | 0.961 |
| SVM | 0.944 |

## Technologies Used

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- XGBoost

## Validation Strategy

- Stratified K-Fold Cross Validation
- GridSearchCV Hyperparameter Tuning

## Results

The MLP neural network achieved the best generalization performance on the Kaggle public leaderboard.

## Future Work

- Deep Learning for raw time-series
- Sequential models (LSTM/CNN)
- Better feature selection

## Author

Juan Pedro
Computer Science Student - UFSCar
