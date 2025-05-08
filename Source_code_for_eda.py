# Step 1: Upload the CSV File
from google.colab import files
uploaded = files.upload()

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 3: Load the dataset
df = pd.read_csv("us_accident_250_samples.csv")  # Ensure this matches the uploaded filename
print("Dataset shape:", df.shape)
print(df.head())

# Step 4: Data Cleaning
df = df.drop_duplicates()
df = df.dropna(subset=['Start_Time', 'Weather_Condition', 'Severity'])
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Step 5: Feature Engineering
df['Hour'] = df['Start_Time'].dt.hour
df['is_peak_hour'] = df['Hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 18 else 0)
df['is_weekend'] = df['Start_Time'].dt.weekday >= 5
df['bad_weather'] = df['Weather_Condition'].isin(['Rain', 'Snow', 'Fog', 'Thunderstorm']).astype(int)

# Step 6: Prepare Input and Target Variables
X = df[['Hour', 'is_peak_hour', 'is_weekend', 'bad_weather']]
y = df['Severity']  # Optionally convert to binary: y = (df['Severity'] >= 3).astype(int)

# Step 7: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Predict and Evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Visualization
plt.figure(figsize=(10, 5))
sns.countplot(x='Hour', data=df)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.show()
