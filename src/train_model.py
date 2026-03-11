import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data

data = preprocess_data("../dataset/insurance_fraud.csv")

X = data.drop("fraud_reported", axis=1)
y = data["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open("../model/fraud_model.pkl", "wb"))

print("Model trained and saved successfully")
