# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An end-to-end Machine Learning and Data Analytics project that analyzes historical retail sales data, forecasts future sales, detects anomalies, segments product demand, and presents insights through an interactive Streamlit dashboard.

---

## 🌐 Live Demo

**Streamlit App:**  
https://salesforecasting-aryanparashar.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/Arceus-Legendary/SalesForecasting_AryanParashar

---

## 📌 Project Overview

This project was developed to help businesses make data-driven decisions for inventory management, sales planning, and demand forecasting.

Using four years of historical Superstore sales data, the system:

- Performs exploratory data analysis (EDA)
- Identifies sales trends and seasonality
- Forecasts future sales using multiple forecasting models
- Detects unusual sales patterns (anomalies)
- Segments products based on demand characteristics
- Presents all insights through an interactive Streamlit dashboard

---

# 🚀 Features

### 📈 Sales Analysis
- Sales trend analysis
- Yearly and monthly sales visualization
- Regional sales comparison
- Category-wise performance analysis

### 🔮 Sales Forecasting
- SARIMA Forecasting
- Facebook Prophet Forecasting
- XGBoost Forecasting
- Model comparison using:
  - MAE
  - RMSE
  - MAPE

### 🚨 Anomaly Detection
- Isolation Forest
- Z-Score Detection
- Weekly anomaly visualization
- Business interpretation of anomalies

### 📦 Product Demand Segmentation
- K-Means Clustering
- PCA Visualization
- Product demand grouping
- Inventory strategy recommendations

### 🌐 Interactive Dashboard
- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

---

# 🛠 Tech Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Time Series Forecasting
- Statsmodels (SARIMA)
- Prophet

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Dashboard
- Streamlit

---

# 📊 Forecasting Models

| Model | MAE | RMSE | MAPE |
|-------|------:|------:|------:|
| SARIMA | 18,031 | 19,009 | 18.97% |
| Prophet | 20,251 | 22,318 | 21.86% |
| **XGBoost** | **14,443** | **17,067** | **14.45%** |

**Best Performing Model:** XGBoost

---

# 📁 Project Structure

```text
SalesForecasting_AryanParashar/

│── app.py
│── analysis.ipynb
│── requirements.txt
│── runtime.txt
│── cleaned_sales.csv
│── train.csv
│── xgboost_sales_model.pkl
│── xgboost_features.csv
│── README.md
│
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Anomaly_Report.py
│   └── 4_Product_Demand_Segments.py
│
└── charts/
```

---

# ▶️ Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/Arceus-Legendary/SalesForecasting_AryanParashar.git

cd SalesForecasting_AryanParashar
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

---

# 📈 Dashboard Pages

### 🏠 Sales Overview
- Total Sales KPI
- Total Orders
- Average Order Value
- Sales Trend
- Region & Category Filters

### 🔮 Forecast Explorer
- Category Forecast
- Region Forecast
- Forecast Horizon Selection
- Forecast Table
- Performance Metrics

### 🚨 Anomaly Report
- Isolation Forest Detection
- Weekly Sales Anomalies
- Anomaly Table

### 📦 Product Demand Segments
- K-Means Clustering
- PCA Visualization
- Demand Segment Table
- Inventory Recommendations

---

# 📌 Business Value

This system helps businesses:

- Improve inventory planning
- Forecast future sales demand
- Detect unusual sales activity
- Reduce stock-outs and excess inventory
- Support data-driven supply chain decisions

---

# 📄 Project Deliverables

- ✔ Data Cleaning & Feature Engineering
- ✔ Exploratory Data Analysis
- ✔ Time Series Analysis
- ✔ Sales Forecasting
- ✔ Product Demand Segmentation
- ✔ Anomaly Detection
- ✔ Interactive Streamlit Dashboard
- ✔ Executive Business Report

---

# 👨‍💻 Author

**Aryan Parashar**

B.Tech Artificial Intelligence Student

Zakir Husain College of Engineering & Technology (ZHCET)

Aligarh Muslim University

---

## ⭐ If you found this project interesting, consider giving it a star on GitHub!