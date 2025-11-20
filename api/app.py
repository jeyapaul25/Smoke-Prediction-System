from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder="../templates")

# ----------- TRAIN SIMPLE MODEL -----------
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.uniform(25, 40, 200),
    'humidity': np.random.uniform(40, 90, 200),
    'time_of_day': np.random.randint(0, 24, 200)
})

data['smoke_ppm'] = (
    0.8 * data['temperature']
    - 0.5 * data['humidity']
    + 2 * np.sin(data['time_of_day']/3)
    + np.random.normal(0, 3, 200)
)

X = data[['temperature', 'humidity', 'time_of_day']]
y = data['smoke_ppm']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------- FLASK ROUTES -----------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    time_of_day = int(request.form['time_of_day'])

    predicted_smoke = model.predict([[temperature, humidity, time_of_day]])[0]

    if predicted_smoke > 50:
        action = "⚠️ High Pollution! Activate Buzzer and Start Air Purifier."
    elif predicted_smoke > 30:
        action = "⚠️ Moderate Pollution! Increase Ventilation."
    else:
        action = "✅ Air Quality is Good."

    return render_template(
        "index.html",
        prediction=f"{predicted_smoke:.2f} ppm",
        action=action
    )


# if __name__ == "__main__":
#     app.run(debug=True)
