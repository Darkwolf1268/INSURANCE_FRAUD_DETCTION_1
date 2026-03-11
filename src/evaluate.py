import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

model = pickle.load(open("../model/fraud_model.pkl","rb"))

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
