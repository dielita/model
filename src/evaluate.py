import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.metrics import accuracy_score

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

dataset_name = params["data"]["name"]
features = params["features"]["columns"]
model_type = params["model"]["type"]

def get_exp_id():
    return os.getenv("DVC_EXP_NAME", "noexp")

exp_id = get_exp_id()

# Determine model name
model_filename = f"{model_type}--{dataset_name}--{exp_id}.pkl"
model_path = os.path.join("models", model_filename)

model = joblib.load(model_path)

# Load data again
df = pd.read_csv("data/titanic.csv")

y = df["Survived"]
X = pd.get_dummies(df[features])

preds = model.predict(X)

acc = accuracy_score(y, preds)

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics_path = "metrics/evaluate.json"

with open(metrics_path, "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

print(f"Saved metrics: {metrics_path}")