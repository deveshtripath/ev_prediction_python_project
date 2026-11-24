"""Rebuild the files the app needs that the Kaggle export did not include.

    python prepare_data.py path/to/xjtu-2024-cycles.csv

Repeats the notebook's filtering and battery-wise split exactly, then writes:
  battery_health_project/models/scaler.pkl   StandardScaler fitted on the training cells
  data/test_predictions.csv                  held-out test cycles + each model's prediction
  data/train_feature_ranges.csv              min/max of each feature in training data
  data/soh_curves.csv                        SOH per cycle for every cell (every 3rd row)
  battery_health_project/plots/eda_overview.png, correlation_heatmap.png  (from evprediction.ipynb)
"""
import base64
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(ROOT, "battery_health_project")
DATA = os.path.join(ROOT, "data")
FEATURES = ["cycle", "charge_ah", "charge_ah_reported", "v_max", "v_min", "duration_s"]
MODELS = {"Linear Regression": ("linear_regression.pkl", True), "SVR": ("svr.pkl", True),
          "XGBoost": ("xgboost.pkl", False), "Random Forest": ("random_forest.pkl", False)}


def main(csv_path):
    df = pd.read_csv(csv_path)
    test_cap = df["label"].str.contains("test capacity", case=False, na=False)
    one_c = df["label"].str.contains("1C discharge", case=False)
    keep = ((df["batch"].isin(["Batch-1", "Batch-2"]) & one_c & ~test_cap)
            | ((df["batch"] == "Batch-3") & (df["label"] == "2C charge and 1C discharge"))
            | (df["batch"].isin(["Batch-4", "Batch-5"]) & test_cap & one_c))
    s = df.loc[keep].copy()
    s["SOH"] = s["discharge_ah"] / 2.0 * 100

    X, y = s[FEATURES], s["SOH"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=s["cell"]))
    scaler = StandardScaler().fit(X.iloc[tr])
    joblib.dump(scaler, os.path.join(PROJ, "models", "scaler.pkl"))

    out = s.iloc[te][["batch", "cell"] + FEATURES + ["SOH"]].rename(columns={"SOH": "actual_soh"})
    for name, (fname, scaled) in MODELS.items():
        model = joblib.load(os.path.join(PROJ, "models", fname))
        Xte = X.iloc[te]
        pred = model.predict(scaler.transform(Xte) if scaled else Xte)
        out[name] = np.round(pred, 4)
        print(f"{name:18s} MAE {mean_absolute_error(y.iloc[te], pred):.4f}  "
              f"RMSE {mean_squared_error(y.iloc[te], pred) ** 0.5:.4f}  R2 {r2_score(y.iloc[te], pred):.4f}")

    os.makedirs(DATA, exist_ok=True)
    out.to_csv(os.path.join(DATA, "test_predictions.csv"), index=False)
    X.iloc[tr].describe().T[["min", "max", "mean"]].to_csv(os.path.join(DATA, "train_feature_ranges.csv"))
    s["split"] = "train"
    s.iloc[te, s.columns.get_loc("split")] = "test"
    s[["batch", "cell", "split", "cycle", "SOH"]].iloc[::3].to_csv(os.path.join(DATA, "soh_curves.csv"), index=False)

    nb_path = os.path.join(ROOT, "evprediction.ipynb")
    if os.path.exists(nb_path):
        nb = json.load(open(nb_path, encoding="utf-8"))
        for cell_idx, name in [(4, "eda_overview"), (5, "correlation_heatmap")]:
            for o in nb["cells"][cell_idx].get("outputs", []):
                if "image/png" in o.get("data", {}):
                    with open(os.path.join(PROJ, "plots", f"{name}.png"), "wb") as f:
                        f.write(base64.b64decode(o["data"]["image/png"]))
    print(f"Done. {len(tr)} train / {len(te)} test cycles, {s.iloc[te]['cell'].nunique()} test cells.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python prepare_data.py path/to/xjtu-2024-cycles.csv")
    main(sys.argv[1])
