import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# 1. Load Dataset
df = pd.read_csv("car_purchasing.csv", encoding='ISO-8859-1')
df.drop(['customer name', 'customer e-mail', 'country'], axis=1, inplace=True)

# 2. Clean & Check
print("Missing values:\n", df.isnull().sum())

# 3. Feature Scaling
scaler = StandardScaler()
scaled_cols = ['age', 'annual Salary', 'credit card debt', 'net worth']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# 4. Marketing-Relevant Feature Engineering
df['debt_ratio'] = df['credit card debt'] / (df['annual Salary'] + 1e-9)
df['wealth_ratio'] = df['net worth'] / (df['annual Salary'] + 1e-9)

# 5. Correlation Heatmap (to identify potential marketing levers)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Marketing Insights)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# 6. Define Target (sales) and Features
X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

print("\nModel Performance (Target: Sales Growth):\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))  # Fixed for older sklearn
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# 8. Hyperparameter Tuning (on Random Forest for marketing optimization)
print("\nTuning Random Forest for Best Marketing Strategy Forecasting...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, scoring='neg_mean_squared_error', cv=3)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
best_preds = best_rf.predict(X_test)
print(f"Optimized Random Forest R2 Score: {r2_score(y_test, best_preds):.3f}")

# 9. Feature Importance (to inform marketing budget allocation)
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title='Marketing Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png")

# 10. Marketing Strategy Insights
print("\nMarketing Insights for Sales Growth Optimization:")
top_features = importances.sort_values(ascending=False).head(3)
for feat, score in top_features.items():
    print(f"• Focus on '{feat}' — it strongly influences sales (importance: {score:.2f})")

print("\nSales forecasting complete. Check visuals: correlation_heatmap.png, feature_importance.png")
