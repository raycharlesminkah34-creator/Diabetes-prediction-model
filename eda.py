import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#_EXPLORATORY DATA ANALYSIS
df = pd.read_csv("diabetes.csv")

zero_cols = ["Glucose", "BloodPressure", "SkinThickness","Insulin", "BMI", "Age"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

df.drop(columns=["Insulin"], inplace=True)
imputer = SimpleImputer(strategy="median")
df[["Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"]] = imputer.fit_transform(df[["Glucose", "BloodPressure", "SkinThickness", "BMI", "Age"]])

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    #print(f"{col}: {len(outliers)} outliers | lower bound: {lower:.2f} | upper bound: {upper:.2f}")

for col in ["SkinThickness", "BloodPressure"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df[col] = df[col].clip(lower, upper)
"""
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
"""

#Model training process 
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)

