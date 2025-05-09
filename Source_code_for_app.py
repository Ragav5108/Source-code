import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
df = pd.read_csv("/storage/emulated/0/Download/us_accident_250_samples.csv")
df = df[['Severity', 'Weather_Condition', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']]
df.dropna(inplace=True)

le = LabelEncoder()
df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])

X = df.drop('Severity', axis=1)
y = df['Severity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

severity_description = {
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Severe"
}

st.title("Accident Severity Predictor")

weather = st.selectbox("Weather Condition", le.classes_)
temp = st.number_input("Temperature (°F)", step=0.1)
humidity = st.number_input("Humidity (%)", step=0.1)
visibility = st.number_input("Visibility (mi)", step=0.1)
wind = st.number_input("Wind Speed (mph)", step=0.1)

if st.button("Predict"):
    try:
        weather_encoded = le.transform([weather])[0]
        input_data = scaler.transform([[weather_encoded, temp, humidity, visibility, wind]])
        prediction = int(model.predict(input_data)[0])
        desc = severity_description.get(prediction, "Unknown")
        st.success(f"Severity Level {prediction} - {desc}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Optional: Show data and chart
st.subheader("Sample Data")
st.dataframe(df.head())

# Optional: Add charts
import plotly.express as px
st.plotly_chart(px.scatter(df, x='Temperature(F)', y='Humidity(%)', color='Severity', title="Temperature vs Humidity"))
st.plotly_chart(px.pie(df, names='Severity', title="Severity Distribution"))
