import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
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

#print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
#print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#BALANCED RF MODEL
rf_model_balance = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_balance.fit(X_train, y_train)

y_pred_balanced = rf_model_balance.predict(X_test)
#print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced)}")
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
#print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

#XGBOOST MODEL - TUNED
xgb2_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    scale_pos_weight=500/268,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb2_model.fit(X_train, y_train)
y_pred_xgb2 = xgb2_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb2)}")
#print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb2))

#CROSS_VAL
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='recall')
print("CV Scores:", cv_scores)
print("Mean recall:", cv_scores.mean().round(3))
print("Std:", cv_scores.std().round(3))

#FEATURE IMPORTANCE VISUALS
#importance = pd.Series(xgb2_model.feature_importances_,index=X_train.columns)
#importance.sort_values().plot(kind='barh', figsize=(8,5), title="XGBOOST FEATURE IMPORTANCE")
#plt.tight_layout()
#plt.show()

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=50,
    scoring='recall',
    cv=5,
    random_state=42           
)

search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV Recall:", search.best_score_.round(3))

best_xgb = search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
print(classification_report(y_pred_best, y_test))


import joblib
joblib.dump(xgb2_model, 'diabetes_xgb_best.pkl')

