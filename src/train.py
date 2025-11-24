import pandas as pd
import joblib
import yaml
import os
from sklearn.ensemble import RandomForestClassifier

# ---------- Load params ----------
with open("params.yaml") as f:
    params = yaml.safe_load(f)

def get_exp_id():
    return os.getenv("DVC_EXP_NAME", "noexp")

dataset_name = params["data"]["name"]
features = params["features"]["columns"]
model_params = params["model"]

exp_id = get_exp_id()

# ---------- Load dataset ----------
df = pd.read_csv("data/titanic.csv")  # импортированный .dvc развернёт CSV

y = df["Survived"]
X = pd.get_dummies(df[features])

# ---------- Train Model ----------
model = RandomForestClassifier(
    n_estimators=model_params["n_estimators"],
    max_depth=model_params["max_depth"],
    random_state=model_params["random_state"]
)

model.fit(X, y)

# ---------- Save Model ----------
model_filename = f"{model_params['type']}--{dataset_name}--{exp_id}.pkl"
save_path = os.path.join("models", model_filename)
os.makedirs("models", exist_ok=True)

joblib.dump(model, save_path)

print(f"Saved model: {save_path}")