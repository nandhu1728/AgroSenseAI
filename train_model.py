import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode crop names
le = LabelEncoder()
data["crop"] = le.fit_transform(data["crop"])

# Features and target
X = data[["moisture","N","P","K","temperature","humidity","light"]]
y = data["crop"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "crop_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model trained successfully!")