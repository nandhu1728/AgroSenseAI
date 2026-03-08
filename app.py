from flask import Flask, render_template, request
import joblib
import random

app = Flask(__name__)

# Load ML model
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# Store last 5 predictions
history = []

# Crop Name Translation
crop_translation = {
    "Rice": "அரிசி",
    "Millet": "கம்பு",
    "Groundnut": "வேர்க்கடலை",
    "Sugarcane": "கரும்பு",
    "Maize": "மக்காச்சோளம்"
}

# 🌍 Language Dictionary
translations = {
    "en": {
        "title": "AI based Crop Recommendation System",
        "simulate": "Generate Sensor Data",
        "analyze": "Analyze Soil",
        "recommended": "Recommended Crop",
        "soil_index": "Soil Health Index",
        "poor": "Poor",
        "moderate": "Moderate",
        "healthy": "Healthy",
        "fertilizer": "Recommended Fertilizer",
        "no_fert": "No additional fertilizer needed",
        "recent": "Recent Predictions",
        "invalid": "Invalid Input",
        "confidence": "Confidence"
    },
    "ta": {
        "title": "மண் தரத்தை அடிப்படையாகக் கொண்ட AI பயிர் பரிந்துரை அமைப்பு",
        "simulate": "சென்சார் தரவு உருவாக்கவும்",
        "analyze": "மண் பகுப்பாய்வு செய்யவும்",
        "recommended": "பரிந்துரைக்கப்படும் பயிர்",
        "soil_index": "மண் ஆரோக்கிய குறியீடு",
        "poor": "மோசமானது",
        "moderate": "மிதமானது",
        "healthy": "ஆரோக்கியமானது",
        "fertilizer": "பரிந்துரைக்கப்படும் உரம்",
        "no_fert": "கூடுதல் உரம் தேவையில்லை",
        "recent": "சமீபத்திய பரிந்துரைகள்",
        "invalid": "தவறான உள்ளீடு",
        "confidence": "நம்பிக்கை"
    }
}

# 🏠 Home Route
@app.route("/")
def home():
    lang = request.args.get("lang", "en")
    return render_template(
        "index.html",
        t=translations[lang],
        lang=lang,
        simulated=None,
        result=None,
        soil=None,
        fertilizer=None,
        history=history
    )

# 🎛 Simulation Route
@app.route("/simulate")
def simulate():
    lang = request.args.get("lang", "en")

    simulated_values = {
        "moisture": random.randint(30, 80),
        "N": random.randint(30, 90),
        "P": random.randint(20, 60),
        "K": random.randint(20, 60),
        "temperature": random.randint(20, 35),
        "humidity": random.randint(40, 90),
        "light": random.randint(500, 1200)
    }

    return render_template(
        "index.html",
        t=translations[lang],
        lang=lang,
        simulated=simulated_values,
        result=None,
        soil=None,
        fertilizer=None,
        history=history
    )

# 🤖 Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    lang = request.form.get("lang", "en")

    try:
        values = [
            float(request.form["moisture"]),
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["light"])
        ]

        prediction = model.predict([values])
        probabilities = model.predict_proba([values])

        crop = le.inverse_transform(prediction)[0]

        # Convert crop name to Tamil if Tamil selected
        if lang == "ta":
            crop = crop_translation.get(crop, crop)

        confidence = round(max(probabilities[0]) * 100, 2)

        result_text = f"{crop} ({translations[lang]['confidence']}: {confidence}%)"

        moisture, N, P, K, temp, humidity, light = values

        health_score = round((moisture + N + P + K + humidity) / 5, 2)

        if health_score < 40:
            soil_status = translations[lang]["poor"]
        elif health_score < 70:
            soil_status = translations[lang]["moderate"]
        else:
            soil_status = translations[lang]["healthy"]

        soil_text = f"{translations[lang]['soil_index']}: {health_score}/100 ({soil_status})"

        fertilizer_list = []

        if N < 50:
            fertilizer_list.append("Urea" if lang == "en" else "யூரியா")
        if P < 40:
            fertilizer_list.append("DAP" if lang == "en" else "டி.ஏ.பி")
        if K < 40:
            fertilizer_list.append("Potash" if lang == "en" else "பொட்டாஷ்")

        if not fertilizer_list:
            fertilizer_text = translations[lang]["no_fert"]
        else:
            fertilizer_text = ", ".join(fertilizer_list)

        # Save history
        history.insert(0, result_text)
        if len(history) > 5:
            history.pop()

        return render_template(
            "index.html",
            t=translations[lang],
            lang=lang,
            simulated=None,
            result=result_text,
            soil=soil_text,
            fertilizer=fertilizer_text,
            history=history
        )

    except:
        return render_template(
            "index.html",
            t=translations[lang],
            lang=lang,
            simulated=None,
            result=translations[lang]["invalid"],
            soil=None,
            fertilizer=None,
            history=history
        )

if __name__ == "__main__":
    app.run(debug=True)