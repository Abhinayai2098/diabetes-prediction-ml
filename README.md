# 🩺 Diabetes Prediction Using Machine Learning

## Project Overview

This project focuses on analyzing health-related data and developing machine learning classification models to predict diabetes status based on demographic and medical indicators.

The project follows a complete data science workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and comparison.

**Target Variable:**
- `0` – No Diabetes
- `1` – Diabetes

## 🎯 Project Objectives

1. Analyze the distribution of diabetes cases in the dataset.
2. Investigate the relationship between diabetes and demographic factors such as age and gender.
3. Examine diabetes patterns across different smoking history categories.
4. Study the influence of BMI, hypertension, and heart disease indicators on diabetes-related patterns.
5. Compare multiple machine learning classification algorithms to evaluate their predictive performance.

## 📂 Dataset Description

The dataset contains demographic and health-related information.

| Feature | Description |
|---|---|
| gender | Gender of the individual |
| age | Age of the individual |
| hypertension | Hypertension status |
| heart_disease | Heart disease status |
| smoking_history | Smoking history category |
| bmi | Body Mass Index |
| HbA1c_level | HbA1c level |
| blood_glucose_level | Blood glucose level |
| diabetes | Target variable |

## 🛠️ Technologies and Libraries Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning and model evaluation

## ⚙️ Project Workflow

### 1. Data Preprocessing

- Loaded and explored the dataset.
- Checked dataset structure and data types.
- Standardized categorical values.
- Converted columns into appropriate data types.
- Identified and removed duplicate records.
- Handled missing numerical values using median imputation.
- Handled missing categorical values using mode imputation.
- Detected outliers using the Interquartile Range (IQR) method.

### 2. Exploratory Data Analysis (EDA)

Performed statistical and visual analysis using:

- Correlation analysis
- Covariance analysis
- Skewness and kurtosis analysis
- Diabetes distribution analysis
- Age distribution analysis
- Gender-wise diabetes analysis
- Smoking history analysis
- HbA1c and blood glucose comparisons
- BMI distribution analysis
- Correlation heatmap

### 3. Feature Engineering

Created additional features to improve data representation:

- Age Groups
- BMI Categories

Applied categorical encoding using One-Hot Encoding.

### 4. Machine Learning Models

Implemented four classification algorithms:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Linear classification |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Support Vector Machine (SVM) | Margin-based classification |
| Random Forest | Ensemble-based classification |

Applied feature scaling using StandardScaler for Logistic Regression, KNN, and SVM.

Used an 80:20 training-testing split with stratification to preserve class proportions.

### 5. Model Evaluation

Evaluated all four models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Models were compared based on their F1 scores to identify the best-performing model.

Random Forest feature importance was also analyzed to identify influential predictive features.

## 📊 Key Observations

1. The dataset contains demographic and medical indicators associated with diabetes status.
2. HbA1c and blood glucose levels provide important information for distinguishing diabetes categories.
3. Age and BMI help reveal additional patterns in the dataset.
4. Diabetes distribution across gender and smoking history provides useful insights into the data.
5. Random Forest feature importance helps identify which features contribute most to the model's predictions.
6. Comparing precision, recall, and F1 score provides a more comprehensive evaluation than accuracy alone.

## 🏁 Conclusion

This project demonstrates the application of machine learning techniques to diabetes classification.

Four classification algorithms were implemented, trained, and evaluated using multiple performance metrics.

The model with the highest F1 score was selected as the best-performing model among the four evaluated algorithms.

The project provided practical experience in data preprocessing, exploratory data analysis, feature engineering, classification algorithms, and model evaluation.

## ⚠️ Limitations

- The dataset is observational and may not represent all populations.
- The relationship between input features and diabetes status does not establish causation.
- HbA1c and blood glucose may overlap with the criteria used to assign diabetes labels, potentially introducing target leakage.
- The model has not been clinically validated and should not be used for medical diagnosis.
- Independent validation is required before any real-world application.

## 👨‍💻 Author

**Abhinayai Sura**

Data Science | Machine Learning | Python

## 🙏 Acknowledgement

Special thanks to **Dr. Mrinalini Rana** for her guidance and support throughout this project.

---

⭐ If you find this project useful, feel free to explore the repository and share your feedback.
