import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load Dataset
data = pd.read_csv("housing.csv")   # adjust path if needed

# Target column (as seen in your code it's 'median_house_value')
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# 2. Load Pipeline & Model
pipeline = joblib.load("pipeline.pkl")
model = joblib.load("model.pkl")

# 3. Preprocess Data & Split
X_processed = pipeline.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results 📊")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.2f}")
