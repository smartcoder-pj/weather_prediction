import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import pickle

# Load dataset
data = pd.read_csv("weather.csv")

# Features
X = data[["Temperature", "Humidity", "WindSpeed"]]

# Target
y = data["Weather"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM model
model = SVC(kernel="rbf")
model.fit(X, y)

# Save model, scaler and encoder
pickle.dump(model, open("weather_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model trained successfully!")
