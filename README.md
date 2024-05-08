# Student Dropout Prediction

## Overview
This project aims to predict student dropout using machine learning techniques, involving data analysis, preprocessing, feature selection, model training, and evaluation. The dataset contains information about students, including demographics, academic performance, and other relevant factors. The code uses Streamlit to create a web interface for the dropout prediction model, loading a pre-trained Random Forest Classifier model and a Label Encoder for preprocessing categorical variables. Users can input student details and select options for categorical variables, with the model predicting whether the student is likely to drop out or not based on the input.

## Getting Started
To get started with the project, you'll need Python, Streamlit and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- yellowbrick

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost yellowbrick
```

## Usage
1. Clone the repository

2. Navigate to the project directory:
```bash
cd student-dropout-prediction
```

3. Run the Jupyter notebook to see the analysis, preprocessing, model training, and evaluation.

## Dataset
The dataset contains `395` records with `31` features. It includes information about students such as school, sex, age, family size, academic performance, etc.

## Models Used
- Logistic Regression
- Naive Bayes (BernoulliNB)
- K Nearest Neighbors
- Decision Tree
- Support Vector Machine
- Random Forest
- XGBoost

## Results
The Random Forest Classifier performed the best with an accuracy of 81% and an F1 score of 0.84 (approximately).

## Conclusion
The project successfully predicted student dropout using machine learning. The Random Forest Classifier showed the best performance among the models tested with 81% accuracy.
```
