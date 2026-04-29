import joblib

class Detector:
    def __init__(self):
        self.model = joblib.load("backend/detector/mlp_model.joblib")

    def predict(self, features):
        pred = self.model.predict([features])[0]
        prob = self.model.predict_proba([features])[0]

        return pred, prob