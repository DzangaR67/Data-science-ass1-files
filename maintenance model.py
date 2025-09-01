# maintenance_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r'C:/Users/USER/Downloads/Data Science ass1/Question 1 datasets.csv')



# Feature Engineering
df['Vibration_Runtime'] = df['Vibration'] * df['Runtime']
df['Temp_Pressure'] = df['Temperature'] * df['Pressure']

# Define features and target
features = ['Temperature', 'Vibration', 'Pressure', 'Runtime', 'Vibration_Runtime', 'Temp_Pressure']
X = df[features]
y = df['Days to Failure']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R² Score: {r2:.2f}")

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"Cross-Validated RMSE: {-cv_scores.mean():.2f}")

# Plot feature importance
importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

#Actual vs. Predicted Scatter Plot
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Days to Failure")
plt.ylabel("Predicted Days to Failure")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()


#Risiduals plot
residuals = y_test - y_pred
plt.hist(residuals, bins=30, edgecolor='black')
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
