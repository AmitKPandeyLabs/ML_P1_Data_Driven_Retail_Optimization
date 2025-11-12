# Data-Driven Retail Optimization: Superstore Sales Prediction

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/XGBoost-1.7%2B-darkgreen.svg?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-purple.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>

This repository contains an end-to-end machine learning project on "Data-Driven Retail Optimization." The project focuses on analyzing a large retail dataset to predict sales and uncover actionable business insights.

## 🎯 Project Overview & Business Problem

The retail industry is evolving rapidly, and businesses must leverage data to remain competitive. This project focuses on using data-driven approaches to enhance profitability and streamline operations for a large superstore.

The primary business motivations for this project were:
* **Business Insights:** Analyze sales data to uncover trends and customer behaviors that influence business strategies.
* **Profitability:** Understand the complex relationship between discounts, sales, and profits to optimize pricing and promotional strategies.
* **Operational Efficiency:** Streamline logistics and supply chain processes by analyzing shipping data.

The technical goal was to preprocess a complex dataset, perform deep exploratory data analysis, and implement several regression models to accurately predict `Sales`.

## 📊 Dataset

The dataset is a comprehensive "Superstore" dataset containing transaction-level data for a retail chain. It includes key features such as:
* `Order Date` & `Ship Date`
* `Ship Mode`
* `Segment` & `Category` / `Sub-Category`
* `Sales` (Target Variable)
* `Quantity`
* `Discount`
* `Profit`

## ⚙️ Methodology

The project followed a structured machine learning pipeline:

### 1. Data Preprocessing
A thorough preprocessing phase was conducted to prepare the data for modeling.
* **Data Cleaning:** Cleaned the data by dropping rows with NaT (Not a Time) values in date columns.
* **Feature Engineering:** Removed non-essential features like `Order Date` and `Ship Date` that were not used in the final model.
* **Encoding:** Applied **Label Encoding** to transform all object-type (categorical) features (like `Category`, `Sub-Category`, `Segment`, etc.) into a machine-readable numerical format.
* **Data Splitting:** Defined the feature matrix (X) and the target variable (y - `Sales`) and split the data into training and test sets for model evaluation.
* **Standardization:** Standardized all feature data using `StandardScaler` to ensure all features have a mean of 0 and a standard deviation of 1. This prevents models from being biased by features with different scales.

### 2. Exploratory Data Analysis (EDA)
A deep exploratory data analysis was conducted to understand distributions, relationships, and outliers. Here are the key findings from the analysis:
