import joblib

model = joblib.load("models/logistic_regression/model.pkl")
vectorizer = joblib.load("models/logistic_regression/vectorizer.pkl")

def predict_logistic(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return ["Negative", "Neutral", "Positive"][pred]
