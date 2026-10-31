# DIABETES PREDICTION PROJECT
# Data Preprocessing, Exploratory Data Analysis and Machine Learning
# 0 = No Diabetes, 1 = Diabetes

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)


# LOAD THE DATASET
df = pd.read_csv("C:/Users\scs73\Downloads\diabetes_prediction_dataset.csv")

# Display the first five records
print("First five records:")
print(df.head(5))

# Display the last five records
print("\nLast five records:")
print(df.tail(5))

# Display dataset information, including columns, data types and non-null counts
print("\nDataset information:")
df.info()

# Display descriptive statistics for numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Display the original number of rows and columns
print("\nOriginal dataset shape:", df.shape)


# DATA PREPROCESSING AND TYPE CONVERSION


# Remove extra spaces and standardize categorical text values
df["gender"] = df["gender"].str.strip().str.title()
df["smoking_history"] = df["smoking_history"].str.strip().str.lower()

# Convert numerical columns to appropriate numeric data types
numerical_columns = [
    "age", "hypertension", "heart_disease",
    "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
]

for column in numerical_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# DUPLICATE VALUE DETECTION AND REMOVAL


# Count and display duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

print("Dataset shape after removing duplicates:", df.shape)

# MISSING VALUE ANALYSIS AND HANDLING


# Display the number of missing values before handling
print("\nMissing values before handling:")
print(df.isnull().sum())

# Replace missing numerical values with the median of each column
numerical_columns = df.select_dtypes(include="number").columns

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# Replace missing categorical values with the mode of each column
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

# Display the number of remaining missing values after handling
print("\nMissing values after handling:")
print(df.isnull().sum())

# CORRELATION AND COVARIANCE ANALYSIS


# Calculate correlations between numerical columns
# Correlation measures the strength and direction of linear relationships
print("\nCorrelation matrix:")
print(df.corr(numeric_only=True).round(2))

# Calculate covariance between numerical columns
# Covariance indicates whether two variables tend to vary together
print("\nCovariance matrix:")
print(df.cov(numeric_only=True).round(2))


# SKEWNESS AND KURTOSIS ANALYSIS


# Calculate skewness for numerical columns
print("\nSkewness of numerical columns:")
print(df.select_dtypes(include="number").skew().round(2))

# Calculate kurtosis for numerical columns
print("\nKurtosis of numerical columns:")
print(df.select_dtypes(include="number").kurt().round(2))


# OUTLIER DETECTION USING THE IQR METHOD

# Select numerical input columns automatically, excluding the target
outlier_columns = df.select_dtypes(include="number").columns.drop("diabetes")

# Detect values outside the 1.5 x IQR boundaries in each numerical column
print("\nOutlier detection using IQR:")

for column in outlier_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate the interquartile range
    IQR = Q3 - Q1

    # Identify outliers using the lower and upper boundaries
    outliers = df[
        (df[column] < (Q1 - 1.5 * IQR)) |
        (df[column] > (Q3 + 1.5 * IQR))
    ]

    # Display the number of detected outliers
    print(f"Outliers in {column}: {len(outliers)}")


# EXPLORATORY DATA ANALYSIS AND VISUALIZATION


# VISUALIZATION 1: DISTRIBUTION OF DIABETES

# Plot the number of people with and without diabetes
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="diabetes",
    hue="diabetes",
    palette=["skyblue", "orange"],
    legend=False
)

plt.title("Distribution of Diabetes")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.show()

# VISUALIZATION 2: AGE DISTRIBUTION

# Plot a histogram with a KDE curve to visualize age distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    bins=30,
    kde=True,
    color="cornflowerblue",
    edgecolor="black"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# VISUALIZATION 3: DIABETES BY GENDER

# Compare diabetes cases across gender categories
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="diabetes",
    palette=["skyblue", "orange"]
)

plt.title("Diabetes Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of People")
plt.tight_layout()
plt.show()

# VISUALIZATION 4: DIABETES BY SMOKING HISTORY

# Compare diabetes cases across smoking history categories
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="smoking_history",
    hue="diabetes",
    palette=["skyblue", "orange"]
)

plt.title("Diabetes Distribution by Smoking History")
plt.xlabel("Smoking History")
plt.ylabel("Number of People")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# VISUALIZATION 5: HbA1c LEVEL BY DIABETES STATUS

# Compare HbA1c level distributions for both diabetes categories
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="diabetes",
    y="HbA1c_level",
    hue="diabetes",
    palette=["#66B3FF", "#FF9999"],
    legend=False
)

plt.title("HbA1c Level by Diabetes Status")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("HbA1c Level")
plt.tight_layout()
plt.show()

# VISUALIZATION 6: BLOOD GLUCOSE LEVEL BY DIABETES STATUS

# Compare blood glucose levels for people with and without diabetes
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="diabetes",
    y="blood_glucose_level",
    hue="diabetes",
    palette=["skyblue", "salmon"],
    legend=False
)

plt.title("Blood Glucose Level by Diabetes Status")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Blood Glucose Level")
plt.tight_layout()
plt.show()

# VISUALIZATION 7: BMI DISTRIBUTION BY DIABETES STATUS

# Compare BMI values for both diabetes categories
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="diabetes",
    y="bmi",
    hue="diabetes",
    palette=["#8DD3C7", "#FDB462"],
    legend=False
)

plt.title("BMI by Diabetes Status")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("BMI")
plt.tight_layout()
plt.show()

# VISUALIZATION 8: CORRELATION HEATMAP

# Select all numerical features automatically for correlation analysis
selected_columns = df.select_dtypes(include="number").columns

# Calculate correlations between all numerical columns
corr = df[selected_columns].corr(numeric_only=True)

# Visualize correlations using a heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Diabetes Features")
plt.tight_layout()
plt.show()

# FIVE OBJECTIVES OF THE DIABETES PREDICTION PROJECT


# OBJECTIVE 1: ANALYZE OVERALL DIABETES DISTRIBUTION

# Count people in each diabetes category
print("\nOBJECTIVE 1: Overall Diabetes Distribution")
print(df["diabetes"].value_counts())

# Calculate the percentage of each diabetes category
print("\nDiabetes percentage:")
print(
    (df["diabetes"].value_counts(normalize=True) * 100).round(2)
)

# OBJECTIVE 2: STUDY DIABETES BY GENDER

# Compare diabetes counts for each gender
print("\nOBJECTIVE 2: Diabetes by Gender")
print(pd.crosstab(df["gender"], df["diabetes"]))

# Calculate the diabetes percentage within each gender
print("\nPercentage within each gender:")
print(
    (pd.crosstab(
        df["gender"],
        df["diabetes"],
        normalize="index"
    ) * 100).round(2)
)

# OBJECTIVE 3: INVESTIGATE DIABETES BY SMOKING HISTORY

# Compare diabetes counts for each smoking history category
print("\nOBJECTIVE 3: Diabetes by Smoking History")
print(pd.crosstab(df["smoking_history"], df["diabetes"]))

# Calculate the diabetes percentage within each smoking category
print("\nPercentage within each smoking history group:")
print(
    (pd.crosstab(
        df["smoking_history"],
        df["diabetes"],
        normalize="index"
    ) * 100).round(2)
)

# OBJECTIVE 4: EXAMINE AGE AND BMI PATTERNS

# Calculate average age and BMI for each diabetes category
print("\nOBJECTIVE 4: Average Age and BMI by Diabetes Status")
print(
    df.groupby("diabetes")[["age", "bmi"]]
    .mean().round(2)
)

# OBJECTIVE 5: ANALYZE HEALTH INDICATORS

# Compare hypertension, heart disease, HbA1c and glucose by diabetes status
health_columns = [
    "hypertension", "heart_disease",
    "HbA1c_level", "blood_glucose_level"
]

print("\nOBJECTIVE 5: Health Indicators by Diabetes Status")
print(
    df.groupby("diabetes")[health_columns]
    .mean().round(2)
)

# FEATURE ENGINEERING


# Create age groups from customer age
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=["0-18", "19-30", "31-45", "46-60", "Above 60"],
    include_lowest=True
)

# Create BMI categories
df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    include_lowest=True
)

# Display the newly created features
print("\nNew features created:")
print(["age_group", "bmi_category"])

print("\nDataset shape after feature engineering:", df.shape)


# PREPARE DATA FOR CLASSIFICATION


# Select input features and target variable
X = df.drop("diabetes", axis=1)
Y = df["diabetes"].astype(int)

# Convert categorical columns into dummy variables
X = pd.get_dummies(
    X,
    columns=["gender", "smoking_history", "age_group", "bmi_category"],
    drop_first=True,
    dtype=int
)

# Display target class distribution
print("\nTarget class distribution:")
print(Y.value_counts())

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("\nTraining dataset shape:", X_train.shape)
print("Testing dataset shape:", X_test.shape)

# Standardize the features for Logistic Regression, KNN and SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# CLASSIFICATION MODELS
# ------------------------------------------------------------

# CLASSIFICATION MODEL 1: LOGISTIC REGRESSION

# Create and train the Logistic Regression model
model1 = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model1.fit(X_train_scaled, Y_train)

# Predict diabetes categories
y_pred1 = model1.predict(X_test_scaled)

# Evaluate the Logistic Regression model
print("\nEvaluation parameters for Logistic Regression model")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred1))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred1, zero_division=0))

# CLASSIFICATION MODEL 2: K-NEAREST NEIGHBORS

# Create and train the KNN model
# A small K value keeps prediction practical for this dataset
k = 5
print("\nSelected K:", k)

model2 = KNeighborsClassifier(n_neighbors=k)

model2.fit(X_train_scaled, Y_train)

# Predict diabetes categories
y_pred2 = model2.predict(X_test_scaled)

# Evaluate the KNN model
print("\nEvaluation parameters for KNN model")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred2))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred2, zero_division=0))

# CLASSIFICATION MODEL 3: SUPPORT VECTOR MACHINE

# Train SVM on a random sample to keep execution time practical
svm_sample_size = min(10000, len(X_train_scaled))

rng = np.random.RandomState(42)

svm_indices = rng.choice(
    len(X_train_scaled),
    size=svm_sample_size,
    replace=False
)

X_svm_train = X_train_scaled[svm_indices]
Y_svm_train = Y_train.iloc[svm_indices]

# Create and train the Support Vector Machine model
model3 = SVC(
    kernel="linear",
    class_weight="balanced"
)

model3.fit(X_svm_train, Y_svm_train)

# Predict diabetes categories
y_pred3 = model3.predict(X_test_scaled)

# Evaluate the Support Vector Machine model
print("\nEvaluation parameters for Support Vector Machine model")
print("SVM training sample size:", svm_sample_size)
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred3))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred3, zero_division=0))

# CLASSIFICATION MODEL 4: RANDOM FOREST

# Create and train the Random Forest model
model4 = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model4.fit(X_train, Y_train)

# Predict diabetes categories
y_pred4 = model4.predict(X_test)

# Evaluate the Random Forest model
print("\nEvaluation parameters for Random Forest model")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred4, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred4, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred4))

print("\nClassification Report:")
print(classification_report(Y_test, y_pred4, zero_division=0))

# Display the most important features identified by Random Forest
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model4.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot the top 10 important features
plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title("Top 10 Features - Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# MODEL ACCURACY COMPARISON


# Store the names of the four classification models
models = [
    "Logistic Regression",
    "KNN",
    "Support Vector Machine",
    "Random Forest"
]

# Display evaluation parameters for Logistic Regression
print("\nMODEL 1: Logistic Regression")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred1, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred1, zero_division=0))

# Display evaluation parameters for KNN
print("\nMODEL 2: KNN")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred2, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred2, zero_division=0))

# Display evaluation parameters for Support Vector Machine
print("\nMODEL 3: Support Vector Machine")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred3, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred3, zero_division=0))

# Display evaluation parameters for Random Forest
print("\nMODEL 4: Random Forest")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, zero_division=0))
print("Recall:", recall_score(Y_test, y_pred4, zero_division=0))
print("F1 Score:", f1_score(Y_test, y_pred4, zero_division=0))

# Create a model comparison table
predictions = [y_pred1, y_pred2, y_pred3, y_pred4]

model_comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": [
        accuracy_score(Y_test, prediction)
        for prediction in predictions
    ],
    "Precision": [
        precision_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ],
    "Recall": [
        recall_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ],
    "F1 Score": [
        f1_score(Y_test, prediction, zero_division=0)
        for prediction in predictions
    ]
})

# Sort models according to F1 score
model_comparison = model_comparison.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nMODEL COMPARISON:")
print(model_comparison.to_string(index=False))

# Identify the best-fit model using F1 score
best_model_name = model_comparison.iloc[0]["Model"]
best_accuracy = model_comparison.iloc[0]["Accuracy"]
best_f1 = model_comparison.iloc[0]["F1 Score"]

print("\nBEST-FIT MODEL:", best_model_name)
print("Best model accuracy:", round(best_accuracy, 4))
print("Best model F1 score:", round(best_f1, 4))

# Plot the F1 score comparison of all four models
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=model_comparison,
    x="Model",
    y="F1 Score",
    hue="Model",
    palette=["skyblue", "orange", "lightgreen", "salmon"],
    legend=False
)

# Display the F1 score value above each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)

plt.title("F1 Score Comparison of Classification Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# KEY FINDINGS


print("\nKEY FINDINGS")

print(
    "1. The dataset contains demographic details and health indicators "
    "such as age, BMI, hypertension, heart disease, HbA1c and blood glucose."
)

print(
    "2. The target variable identifies whether a person has diabetes."
)

print(
    "3. Age, BMI, HbA1c level and blood glucose level can help explain "
    "patterns associated with diabetes in this dataset."
)

print(
    "4. The Random Forest feature importance analysis identifies the features "
    "that contribute most to its predictions."
)

print(
    "5. The four classification models are compared using accuracy, precision, "
    "recall and F1 score."
)

# FINAL CONCLUSION


print("\nFINAL CONCLUSION")

print(
    "Four machine learning models were developed to classify diabetes status: "
    "Logistic Regression, KNN, Support Vector Machine and Random Forest. "
    "The models were evaluated using accuracy, precision, recall and F1 score. "
    "The model with the highest F1 score was selected as the best-fit model."
)


# RECOMMENDATIONS


print("\nRECOMMENDATIONS")

print(
    "1. Use the model as an analytical aid to identify records with a higher "
    "predicted likelihood of diabetes."
)

print(
    "2. Consider age, BMI, HbA1c and blood glucose patterns when interpreting "
    "the dataset."
)

print(
    "3. Evaluate recall and F1 score along with accuracy because missing a "
    "diabetes-positive case can be important."
)

print(
    "4. Investigate the important features identified by the Random Forest model."
)

print(
    "5. Validate the model on independent, representative data before any "
    "real-world or clinical use."
)


# LIMITATION


print("\nLIMITATION")

print(
    "This dataset is observational and its predictions do not establish a "
    "medical diagnosis or prove that a feature causes diabetes. The model "
    "requires external validation before real-world use."
)

print("\nPROJECT EXECUTION COMPLETED.")

