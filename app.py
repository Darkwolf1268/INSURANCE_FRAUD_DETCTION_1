from flask import Flask,request,jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("models/fraud_model.pkl","rb"))

@app.route("/")
def home():
    return "Insurance Fraud Detection API"

@app.route("/predict",methods=["POST"])
def predict():

    data = request.json["features"]

    prediction = model.predict([data])[0]

    if prediction == 1:
        result = "Fraud Claim"
    else:
        result = "Legitimate Claim"

    return jsonify({"Prediction":result})

if __name__=="__main__":
    app.run(debug=True)
