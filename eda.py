import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
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

#BASE MODEL DEFINE
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

#RANDOM FOREST MODEL
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
#print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#BALANCED RF MODEL
rf_model_balance = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_balance.fit(X_train, y_train)

y_pred_balanced = rf_model_balance.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced)}")
#print("\nClassification Report:\n", classification_report(y_test, y_pred_balanced))

#XGBOOST model
xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=500/268,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))



