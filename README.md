# 🚗 CarDekho Used Car Price Prediction (End-to-End ML Pipeline)

An end-to-end Machine Learning web application designed to forecast used car selling prices based on vehicle technical specifications, historical usage, and mechanical attributes. Built using Scikit-Learn regressors and deployed via an interactive web interface.

---

## 📌 Project Overview
Accurately pricing pre-owned vehicles is challenging due to non-linear depreciation factors such as age, accumulated mileage, fuel economy, engine capacity, and ownership history. 

This project implements a complete predictive pipeline—spanning data cleaning, exploratory data analysis (EDA), feature encoding, standardization, multi-model evaluation, and model serialization—to deliver instantaneous, data-driven valuations.

---

## 📊 Dataset & Features
* **Data Source:** CarDekho vehicle listings (`cardekho.csv`)
* **Dataset Shape:** 8,128 rows, 12 feature columns[cite: 1]
* **Target Variable:** `selling_price` (in INR)[cite: 1]
* **Key Features:**
  * `name`: Vehicle brand and model
  * `year`: Manufacturing year
  * `km_driven`: Total kilometers driven[cite: 1]
  * `fuel`: Fuel type (Diesel, Petrol, CNG, LPG)
  * `seller_type`: Seller classification (Individual, Dealer, Trustmark Dealer)
  * `transmission`: Gearbox transmission (Manual, Automatic)
  * `owner`: Ownership hierarchy (First Owner, Second Owner, etc.)
  * `mileage(km/ltr/kg)`: Fuel efficiency[cite: 1]
  * `engine`: Displacement capacity in CC[cite: 1]
  * `max_power`: Maximum engine output in BHP
  * `seats`: Passenger seating capacity

---

## ⚙️ Key Technical Steps

### 1. Data Cleaning & Feature Transformation
* Stripped text measurement units (`kmpl`, `CC`, `bhp`) from numeric columns and cast continuous features to float types.
* Derived vehicle age directly from manufacturing years to better model depreciation trends.
* Managed data distribution skewness and handled missing values across technical metrics.

### 2. Preprocessing & Scaling
* **StandardScaler:** Standardized continuous features (`km_driven`, `mileage`, `engine`, `max_power`) to eliminate scale disparities and stabilize convergence[cite: 1].
* **Categorical Encoding:** Applied `OneHotEncoder` and `LabelEncoder` across multi-class variables (`fuel`, `seller_type`, `transmission`, `owner`)[cite: 1].
* **Leakage Prevention:** Fitted all scalers and encoders strictly on the training partition before transforming the test split[cite: 1].

### 3. Model Training & Evaluation
Evaluated multiple baseline and ensemble regression algorithms:
* Linear Regression (Baseline model)[cite: 1]
* Decision Tree Regressor[cite: 1]
* Random Forest Regressor[cite: 1]
* **Gradient Boosting Regressor (Top Performer: $R^2 \approx 96.01\%$)**[cite: 1]

### 4. Serialization & Deployment
* Serialized the optimized estimator (`best_model.pkl`), feature column mappings, and preprocessing artifacts via `pickle` / `joblib`[cite: 1, 2].
* Built a standalone, interactive web application (Streamlit / Flask) allowing users to supply vehicle parameters and obtain real-time market valuations[cite: 1].

---

## 🛠️ Tech Stack
* **Language:** Python[cite: 1]
* **Data Manipulation & Analysis:** Pandas, NumPy[cite: 1]
* **Data Visualization:** Matplotlib, Seaborn[cite: 1]
* **Machine Learning:** Scikit-Learn (Linear Regression, Decision Trees, Random Forest, Gradient Boosting, Preprocessing, Metrics)[cite: 1]
* **Model Serialization:** Pickle / Joblib[cite: 2]
* **Deployment/UI:** Streamlit / Flask[cite: 1]

---

## 📁 Repository Structure
```text
├── data/
│   └── cardekho.csv                   # Raw vehicle listings dataset
├── notebooks/
│   └── CarDekho_price_prediction.ipynb # Complete EDA, preprocessing, and model training
├── models/
│   ├── best_model.pkl                 # Serialized Gradient Boosting regressor
│   └── scaler.pkl                     # Standard scaler transformer
├── app.py                             # Web application inference script
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Ignored temporary and cache files
└── README.md                          # Project documentation
```
## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Yadavaniket19/Car-price-prediction-.git
cd Car-price-prediction-
```

### 2. Install Dependencies
```bash
pip install pandas matplotlib scikit-learn jupyter streamlit
```
### 3. Run the Jupyter Notebook
```bash
jupyter notebook CarDekho_price_prediction.ipynb
```
### 4. Launch the Web Application
```bash
streamlit run app.py
```
---

## 👤 Author
- **Aniket Yadav**
- GitHub: [@Yadavaniket19](https://github.com/Yadavaniket19)

---
⭐️ If you found this project helpful, please consider giving it a star!


