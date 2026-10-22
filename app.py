# ==========================================
# STEP 1 & 2: Import Libraries & Load Data
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Replace 'insurance.csv' with your actual file path if needed)
# If you don't have the CSV, you can load it from a URL:
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

print("--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())


# ==========================================
# STEP 3: Missing Values & Outliers
# ==========================================
print("\nMissing values in dataset:")
print(df.isnull().sum())
# Dataset typically has no missing values, but if there were, we'd use df.fillna()


# ==========================================
# STEP 4: Feature Engineering & Preprocessing
# ==========================================
# Encode categorical columns
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# One-hot encode 'region'
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=['charges'])
y = df['charges']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# STEP 5 & 6: Model Building & Evaluation
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = []

for name, model in models.items():
    # Train model (using scaled data for SVR/KNN/Linear, unscaled works fine for trees, but let's use scaled for consistency or raw where appropriate)
    # Tree-based models don't strictly require scaling, but scaling is safe for all.
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Overfitting check (if train R2 is significantly higher than test R2)
    overfitting = "Y" if (train_r2 - test_r2) > 0.1 else "N"
    
    results.append({
        "Model": name,
        "Train RMSE": round(train_rmse, 2),
        "Test RMSE": round(test_rmse, 2),
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Overfitting (Y/N)": overfitting
    })

# Convert results to DataFrame for comparison
comparison_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))


# ==========================================
# STEP 7: Hyperparameter Tuning (Random Forest Example)
# ==========================================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print("\n--- Best Parameters for Random Forest ---")
print(grid_search.best_params_)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Title
st.title("Medical Insurance Cost Prediction App")
st.write("Enter the patient details below to predict estimated medical insurance charges.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children/Dependents", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Preprocess inputs to match training format
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Predict Button
if st.button("Predict Insurance Cost"):
    # Load sample data to fit scaler and model quickly for the demo
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)
    
    # Create input array matching columns
    # Columns: age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_northwest, region_southeast, region_southwest]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    st.success(f"Estimated Medical Insurance Charges: ${prediction[0]:,.2f}")
