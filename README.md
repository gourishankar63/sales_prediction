# 🧠 Sales Forecasting for Marketing Strategy Optimization

This project uses machine learning to forecast car purchase amounts based on customer financial data. It helps businesses identify key factors influencing sales to guide marketing and budgeting decisions.

## 📊 Problem Statement

Businesses need to optimize marketing strategies for maximum sales growth. This project builds a predictive model using historical data to:

- Forecast customer purchase amounts
- Identify key financial features influencing purchase behavior
- Provide actionable insights for marketing budget allocation

## 📁 Dataset

- **Source**: Provided CSV file (`car_purchasing.csv`)
- **Fields used**: Gender, Age, Annual Salary, Credit Card Debt, Net Worth, Car Purchase Amount

## 🔍 Approach

1. **Data Cleaning**: Removed personal identifiers, handled missing values (none in this dataset).
2. **Feature Engineering**: Created ratios like `debt_ratio` and `wealth_ratio` for better business interpretability.
3. **EDA**: Visualized feature correlations and patterns.
4. **Modeling**: Compared Linear Regression, Random Forest, and XGBoost.
5. **Evaluation**: Used MAE, RMSE, and R² metrics.
6. **Feature Importance**: Identified top marketing levers for increasing sales.

## 📌 Key Insights

- Customers with higher wealth-to-salary ratio show more likelihood of higher car purchases.
- Age and credit card debt are also impactful.
- Random Forest with tuning provided the best performance.

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python sales_prediction.py
