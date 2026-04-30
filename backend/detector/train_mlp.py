import json
import joblib
from sklearn.neural_network import MLPClassifier

# Load dataset
with open("backend/detector/dataset.json", "r") as f:
    data = json.load(f)

X = [item["features"] for item in data]
y = [item["label"] for item in data]

model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, early_stopping=False)
model.fit(X, y)

joblib.dump(model, "backend/detector/mlp_model.joblib")

print("Model trained on real dataset")