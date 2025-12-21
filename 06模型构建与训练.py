import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import matplotlib.pyplot as plt


# ==============================
# 1. Load feature data
# ==============================
INPUT_JSON = "restaurants_walk_drive_scores.json"  # 你的最终特征文件

rows = []
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

print(f"[INFO] Loaded data: {df.shape}")


# ==============================
# 2. Define features and target
# ==============================

TARGET_COL = "stars"

# 明确列出不用作为特征的字段
EXCLUDE_COLS = {
    "business_id",
    "stars",
    "latitude",
    "longitude"
}

# 输入特征：除去排除字段以外的所有列
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

print(f"[INFO] Number of features: {len(FEATURE_COLS)}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]


# ==============================
# 3. Train / Val / Test split
#    70 / 15 / 15
# ==============================

# 先 split 出 train (70%) 和 temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# 再把 temp 平分为 val / test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print(f"[INFO] Train size: {X_train.shape[0]}")
print(f"[INFO] Val size:   {X_val.shape[0]}")
print(f"[INFO] Test size:  {X_test.shape[0]}")


# ==============================
# 4. Train LightGBM regressor
# ==============================

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]
)


print("[INFO] Training finished")


# ==============================
# 5. Evaluation (RMSE)
# ==============================

y_pred_test = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f"[RESULT] Test RMSE = {rmse:.4f}")


# ==============================
# 6. Plot: Predicted vs True
# ==============================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([1, 5], [1, 5], "r--", linewidth=1)
plt.xlabel("True Stars")
plt.ylabel("Predicted Stars")
plt.title("Predicted vs True Star Ratings")
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================
# 7. Feature importance plot
# ==============================

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": importances
}).sort_values("importance", ascending=False)

TOP_K = 30
top_feat_imp = feat_imp_df.head(TOP_K)

plt.figure(figsize=(8, 6))
plt.barh(
    top_feat_imp["feature"][::-1],
    top_feat_imp["importance"][::-1]
)
plt.xlabel("Feature Importance")
plt.title(f"Top {TOP_K} Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()
