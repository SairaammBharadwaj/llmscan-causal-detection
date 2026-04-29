from sklearn.neural_network import MLPClassifier
import joblib

# Sample training data (you will expand later)
X = [
    [2.4,0.5,3.0,1.5,1.5,50,45,73],
    [1.2,0.3,1.5,0.9,0.6,10,8,12],
    [3.0,0.7,4.0,2.0,2.0,70,60,80],
    [1.5,0.4,2.0,1.0,1.0,15,12,18]
]

# Labels: 0 = normal, 1 = misbehavior
y = [1,0,1,0]

model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500)
model.fit(X, y)

joblib.dump(model, "backend/detector/mlp_model.joblib")

print("Model trained and saved")