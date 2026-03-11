import pickle
from sklearn.metrics import accuracy_score, classification_report

model = pickle.load(open("models/fraud_model.pkl","rb"))

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
