# Ride Sharing Price Prediction — Assignment Write‑Up (Text)

> Replace **[FIGURE]** placeholders with screenshots/plots from your notebook.

## Problem Statement
Build a supervised machine learning model that predicts **ride price (USD)** from ride and contextual features such as **distance, duration, time of day, day of week, weather, temperature, pickup/dropoff location type, vehicle type, and driver rating**.

## Business Objectives
- **Accurate pricing prediction** to improve fare estimates for users.
- **Understand key price drivers** (e.g., distance/duration, surge times, weather, vehicle class).
- **Deploy a simple web interface** for real‑time predictions.

## Dataset Description
- Rows: **100,000**
- Columns: **11** (10 features + 1 target)
- Target: **`price`** (USD)

**Missing values**
- `temperature`: **1,228** missing
- `driver_rating`: **1,272** missing  
(All other columns have no missing values.)

---

# Part 1: Data Exploration and Understanding

## Q1.1 Initial Data Inspection
- The dataset has **100,000 rows × 11 columns**.
- Feature types include **numerical** (distance, duration, hour, day_of_week, temperature, driver_rating) and **categorical** (weather, pickup_location, dropoff_location, vehicle_type).
- Missing data is limited to **temperature** and **driver_rating** (see counts above).
- Basic summary statistics and sample records are shown in **[FIGURE: df.head()]** and **[FIGURE: df.info()]**.

## Q1.2 Target Variable Analysis (Price)
**Summary statistics (price)**
- Mean: **19.585**
- Median: **13.600**
- Std: **18.600**
- Min / Max: **3.000 / 200.000**
- Skewness: **3.025** (right‑skewed)
- Kurtosis: **14.353** (heavy tails)

**Outliers (IQR rule)**
- Q1 = 8.190, Q3 = 24.000, IQR = 15.810
- Lower/Upper bounds = -15.525 / 47.715
- Outlier count ≈ **6953** (values above the upper bound; lower bound is negative so practically no low outliers)

Include:
- **[FIGURE: histogram of price]**
- **[FIGURE: boxplot of price]**

Interpretation:
- Price is **not normally distributed** and shows **high right skew** (likely due to longer rides, premium vehicles, surge hours, severe weather).

## Q1.3 Feature Distribution Analysis
Include:
- **[FIGURE: histograms for numeric features]**
- **[FIGURE: countplots for categorical features]**

Key observations to mention (based on the data ranges/categories):
- `distance_miles` ranges roughly **0.5–35.22** miles.
- `duration_minutes` ranges **3–120** minutes.
- `hour` spans **0–23**; `day_of_week` spans **0–6**.
- Categorical features have limited levels (weather, pickup/dropoff types, vehicle classes).

## Q1.4 Correlation Analysis (Numeric Features)
Using Pearson correlation on numeric fields (with median imputation for missing values):
- Corr(price, duration_minutes) = **0.828**
- Corr(price, distance_miles) = **0.765**
- Other numeric features have very small correlation magnitudes (|corr| < 0.05).

Include:
- **[FIGURE: correlation heatmap]**

Interpretation:
- **Duration and distance** are the strongest linear drivers of price.

---

# Part 2: Exploratory Data Analysis (EDA)

## Q2.1 Price vs Distance Analysis
- A scatter plot shows a **clear positive relationship** between distance and price.
- Correlation (distance vs price) ≈ **0.765**.

Include:
- **[FIGURE: scatter price vs distance + trend line]**

## Q2.2 Temporal Patterns Analysis
**Hourly average price (top hours)**
- Highest average prices occur around:
  - hour 18: **26.601**
  - hour 8: **26.334**
  - hour 19: **26.114**
  - hour 17: **26.052**
  - hour 23: **25.879**

This suggests **rush hour (morning/evening)** and **late night** are associated with higher prices.

**Day of week (0=Mon … 6=Sun)**
- Average prices by day show modest variation; the highest mean is:
  - day 5 (Saturday): **20.417**
  - day 4 (Friday): **20.199**

Include:
- **[FIGURE: boxplot price by hour]**
- **[FIGURE: boxplot price by day_of_week]**

## Q2.3 Categorical Features Impact
**Vehicle type (mean price)**
- Luxury: **31.304**
- Premium: **23.610**
- Standard: **15.731**
- Shared: **11.211**

**Weather (mean price)**
- Snow: **28.097**
- Heavy Rain: **25.476**
- Light Rain: **20.817**
- Clear: **17.881**
- Cloudy: **17.823**

**Pickup/Dropoff**
- Airport pickup mean: **22.175**
- Airport dropoff mean: **22.130**

Include:
- **[FIGURE: boxplots for price by vehicle_type, weather, pickup_location, dropoff_location]**

Interpretation:
- **Premium/Luxury vehicles** have substantially higher prices.
- **Severe weather (snow/heavy rain)** is associated with higher prices (likely demand/supply impacts).
- **Airport rides** tend to be higher-priced on average.

---

# Part 3: Data Preprocessing and Feature Engineering

## Q3.1 Handle Missing Values
- Numerical missing values were imputed with the **median**:
  - `temperature`
  - `driver_rating`
- Categorical features had no missing values; if missing appears, impute with **mode**.

## Q3.2 Encode Categorical Features
- Use **One‑Hot Encoding** for:
  - weather, pickup_location, dropoff_location, vehicle_type
- Use `handle_unknown="ignore"` to avoid inference-time crashes.

## Q3.3 Feature Scaling
- Apply **StandardScaler** to numerical columns (especially helpful for Linear Regression).
- Use a **Pipeline** so scaling is applied consistently to train/test and during Streamlit predictions.

## Q3.4 Feature Engineering (examples)
At least two engineered features were added:
1. `is_weekend` = 1 if `day_of_week` in {5,6}, else 0  
2. `is_rush_hour` = 1 if `hour` in {7,8,9,16,17,18,19}, else 0  
3. (Optional) `speed_mph` = distance_miles / (duration_minutes / 60)

---

# Part 4: Model Development

## Q4.1 Train‑Test Split
- Split: **80% train / 20% test**
- Random seed: **42** for reproducibility

## Q4.3 Linear Regression
- Train a pipeline: preprocessing → LinearRegression
- Evaluate with RMSE, MAE, R²

## Q4.4 Decision Tree Regressor
- Train DecisionTreeRegressor
- Optionally tune: max_depth, min_samples_leaf

## Q4.5 Random Forest Regressor
- Train RandomForestRegressor
- Extract feature importances

## Q4.6 Voting Regressor (Ensemble)
- Combine (Linear Regression + Decision Tree + Random Forest) using **VotingRegressor**
- For regression, voting is the **average of predictions**

---

# Part 5: Model Evaluation and Selection

## Q5.1 Model Comparison
Below are example metrics from a **15,000-row sample** run (your full-data results may differ depending on hyperparameters and compute):

| Model | RMSE | MAE | R² |
|------|------:|----:|---:|
| Linear Regression | 7.947 | 5.336 | 0.821 |
| Decision Tree | 6.868 | 3.895 | 0.866 |
| Random Forest | 5.029 | 2.919 | 0.928 |

Interpretation:
- The Random Forest captured non‑linear relationships best in this sample evaluation.

## Q5.2 Best Model Interpretation
- If Random Forest is best, discuss:
  - top feature importances (usually duration, distance, vehicle_type, hour)
- If Linear Regression is best, discuss coefficients and directionality.

Include:
- **[FIGURE: feature importance plot]**
- **[FIGURE: residual plot for best model]**

---

# Part 6: Streamlit Web Application

## Q6.1 Basic Structure
- Load the saved model pipeline (joblib)
- Build a form for user inputs
- Predict price and show result

## Q6.2 Interactive Interface
- Use dropdowns for categorical fields and sliders/number inputs for numeric fields
- Add validation (e.g., distance > 0)

Include:
- **[FIGURE: screenshot of Streamlit app running]**

---

# Conclusion
The analysis shows price is primarily driven by **duration and distance**, with noticeable increases during **rush hours**, **late night**, **severe weather**, and for **premium vehicle classes**. A tree-based ensemble (e.g., Random Forest) typically performs best because it learns non-linear effects and interactions that linear models miss.

