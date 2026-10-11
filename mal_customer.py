
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_excel("Mall Customers.xlsx")

# ==========================================
# 2. Explore Dataset
# ==========================================

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ==========================================
# 3. Select Input and Output
# ==========================================

X = df[["Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

# Check missing values in selected columns
if X.isnull().any().any() or y.isnull().any():
    raise ValueError(
        "Input or output contains missing values. "
        "Please handle them before training."
    )

# ==========================================
# 4. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split")
print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# 5. Train Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# 6. Prediction on Test Data
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Test Data")
print("--------------------------------")
print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R² Score :", round(r2, 4))

print("\nLinear Regression Equation")
print(
    f"Spending Score = "
    f"{model.intercept_:.2f} + "
    f"({model.coef_[0]:.2f} × Annual Income)"
)

# ==========================================
# 8. Histogram
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Spending Score (1-100)"],
    bins=20,
    color="skyblue",
    edgecolor="black"
)

plt.title("Distribution of Spending Score")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("Distribution of Spending Score")
plt.show()


# ==========================================
# 9. Scatter Plot + Regression Line
# ==========================================

plt.figure(figsize=(8, 5))

# Show actual dataset
plt.scatter(
    X["Annual Income (k$)"],
    y,
    color="blue",
    alpha=0.7,
    label="Actual Data"
)

# Generate sorted income values for the line
X_line = pd.DataFrame({
    "Annual Income (k$)": np.linspace(
        X["Annual Income (k$)"].min(),
        X["Annual Income (k$)"].max(),
        200
    )
})

# Predict scores for the line
y_line = model.predict(X_line)

plt.plot(
    X_line["Annual Income (k$)"],
    y_line,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Annual Income vs Spending Score")
plt.show()

# ==========================================
# 10. Predict Spending Score for New Income
# ==========================================

income_value = 10

new_customer = pd.DataFrame({
    "Annual Income (k$)": [income_value]
})

predicted_score = model.predict(new_customer)

print("\nNew Customer Prediction")
print("--------------------------------")
print(f"Annual Income: ${income_value * 1000:,.0f}")
print(f"Predicted Spending Score: {predicted_score[0]:.2f}")

# ==========================================
# 11. Compare Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual Spending Score": y_test.values,
    "Predicted Spending Score": np.round(y_pred, 2)
})

print("\nActual vs Predicted (First 10 Test Samples)")
print(comparison.head(10).to_string(index=False))
