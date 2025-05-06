# Source-code
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('US_Accidents_Dataset.csv')  # Replace with actual path

# Data Cleaning
df = df.drop_duplicates()
df = df.dropna(subset=['Start_Time', 'Weather_Condition', 'Severity'])

# Feature Engineering
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['is_peak_hour'] = df['Hour'].apply(lambda x: 1 if x in range(7,10) or x in range(16,19) else 0)
df['is_weekend'] = df['Start_Time'].dt.weekday >= 5
df['bad_weather_condition'] = df['Weather_Condition'].isin(['Rain', 'Snow', 'Fog', 'Thunderstorm']).astype(int)

# Encoding categorical features
encoded = pd.get_dummies(df[['Weather_Condition']], drop_first=True)
df = pd.concat([df, encoded], axis=1)

# Select features and target
features = ['Hour', 'is_peak_hour', 'is_weekend', 'bad_weather_condition'] + list(encoded.columns)
X = df[features]
y = df['Severity']  # You can convert Severity into binary if needed

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted'))
    print("Recall:", recall_score(y_test, preds, average='weighted'))
    print("F1 Score:", f1_score(y_test, preds, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Optional: Cross-validation
cv_scores = cross_val_score(models['Random Forest'], X, y, cv=5)
print("\nRandom Forest CV Accuracy:", np.mean(cv_scores))

# Visualization (Example)
plt.figure(figsize=(10,5))
sns.countplot(x='Hour', data=df)
plt.title('Accidents by Hour of Day')
plt.show()
