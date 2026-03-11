import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data.preprocess import preprocess_data
from src.features.feature_engineering import create_features

data = preprocess_data("data/raw/insurance_fraud.csv")

data = create_features(data)

X = data.drop("fraud_reported", axis=1)
y = data["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open("models/fraud_model.pkl", "wb"))

print("Model trained and saved successfully")
