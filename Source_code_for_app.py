from flask import Flask, request, render_template_string
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv("/storage/emulated/0/Download/us_accident_250_samples.csv")
df = df[['Severity', 'Weather_Condition', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']]
df.dropna(inplace=True)

# Encode
le = LabelEncoder()
df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])

X = df.drop('Severity', axis=1)
y = df['Severity']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

severity_description = {
    1: "Minor",
    2: "Moderate",
    3: "Serious",
    4: "Severe"
}

# Plotly visualizations
scatter_fig = px.scatter(df, x='Temperature(F)', y='Humidity(%)', color='Severity',
                         title="Temperature vs Humidity")
heatmap_fig = px.imshow(df.corr(), text_auto=True, title="Correlation Heatmap")
pie_fig = px.pie(df, names='Severity', title="Severity Distribution",
                 labels={'Severity': 'Severity Level'})

scatter_html = pio.to_html(scatter_fig, full_html=False)
heatmap_html = pio.to_html(heatmap_fig, full_html=False)
pie_html = pio.to_html(pie_fig, full_html=False)

# HTML with Bootstrap
html_template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Accident Severity Predictor</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="bg-light">
<div class="container py-5">
  <h2 class="mb-4">Predict Accident Severity</h2>
  <form method="post" class="mb-5">
    <div class="mb-3">
      <label class="form-label">Weather Condition</label>
      <select name="weather" class="form-select">
        {% for w in weathers %}
          <option value="{{ w }}">{{ w }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="mb-3"><label>Temperature (°F)</label><input class="form-control" type="number" name="temp" step="0.1"></div>
    <div class="mb-3"><label>Humidity (%)</label><input class="form-control" type="number" name="humidity" step="0.1"></div>
    <div class="mb-3"><label>Visibility (mi)</label><input class="form-control" type="number" name="visibility" step="0.1"></div>
    <div class="mb-3"><label>Wind Speed (mph)</label><input class="form-control" type="number" name="wind" step="0.1"></div>
    <button type="submit" class="btn btn-primary">Predict</button>
  </form>

  {% if prediction %}
    <div class="alert alert-info"><h4>Prediction:</h4> Severity Level {{ prediction }} - <strong>{{ description }}</strong></div>
  {% endif %}

  <h3>Sample Dataset</h3>
  <div class="table-responsive mb-4">{{ table|safe }}</div>

  <h3>Data Visualizations</h3>
  {{ scatter_plot|safe }}
  {{ heatmap_plot|safe }}
  {{ pie_chart|safe }}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    description = None
    table_html = df.head().to_html(classes="table table-bordered", index=False)

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

    return render_template_string(html_template, weathers=le.classes_, prediction=prediction,
                                  description=description, table=table_html,
                                  scatter_plot=scatter_html, heatmap_plot=heatmap_html, pie_chart=pie_html)

if __name__ == "__main__":
    app.run(debug=True)
