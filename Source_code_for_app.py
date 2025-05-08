from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask
app = Flask(__name__)

# Load dataset
df = pd.read_csv("/storage/emulated/0/Download/us_accident_250_samples.csv")

# Keep only needed columns
df = df[['Severity', 'Weather_Condition', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']]
df.dropna(inplace=True)

# Encode and preprocess
le = LabelEncoder()
df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])

X = df.drop('Severity', axis=1)
y = df['Severity']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Severity descriptions
severity_description = {
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Severe"
}

# HTML Form
html_template = """
<!doctype html>
<title>Traffic Accident Severity Predictor</title>
<h2>Predict Accident Severity</h2>
<form method=post>
  Weather Condition: <select name=weather>
    {% for w in weathers %}
      <option value="{{ w }}">{{ w }}</option>
    {% endfor %}
  </select><br><br>
  Temperature (°F): <input type=number name=temp step="0.1"><br><br>
  Humidity (%): <input type=number name=humidity step="0.1"><br><br>
  Visibility (mi): <input type=number name=visibility step="0.1"><br><br>
  Wind Speed (mph): <input type=number name=wind step="0.1"><br><br>
  <input type=submit value=Predict>
</form>
{% if prediction %}
<h3>Predicted Severity Level: {{ prediction }} - <b>{{ description }}</b></h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    description = None

    if request.method == "POST":
        try:
            weather = le.transform([request.form['weather']])[0]
            temp = float(request.form['temp'])
            humidity = float(request.form['humidity'])
            visibility = float(request.form['visibility'])
            wind = float(request.form['wind'])

            input_data = scaler.transform([[weather, temp, humidity, visibility, wind]])
            prediction = int(model.predict(input_data)[0])
            description = severity_description.get(prediction, "Unknown")
        except:
            prediction = "Error"
            description = "Invalid input or model issue."

    return render_template_string(html_template, weathers=le.classes_, prediction=prediction, description=description)

if __name__ == "__main__":
    app.run(debug=True)
