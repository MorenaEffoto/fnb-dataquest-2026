
live website :  https://fnb-dataquest-2026.onrender.com


# FNB DataQuest 2026 — From Roots to Rise

A clean, explainable credit risk dashboard built using logistic regression and feature engineering.

This project simulates a real banking credit scoring workflow by transforming messy historical loan data into a system that risk analysts and business users can actually understand and trust.

## The Story

Banks make lending decisions every day:
Should this applicant receive a loan or not?

A bad decision increases default risk. Being too conservative means losing good customers.

I built this project to demonstrate how interpretable machine learning can be used in finance instead of relying entirely on black-box models that are difficult to explain in regulated environments.

Starting with a dataset of roughly 120,000 historical loan applications, I went through the full workflow:
- Data cleaning
- Exploratory analysis
- Feature engineering
- Predictive modelling
- Dashboard development
- Deployment

The final result is an interactive credit risk dashboard that allows users to explore default behaviour, simulate lending decisions, and analyse portfolio risk.

## Key Results

- Baseline AUC: 0.68
- Final AUC: 0.797 using fully interpretable methods

## Why Logistic Regression?

A lot of ML projects focus only on maximising accuracy. I wanted to build something closer to what many banks actually use in practice.

This project uses:
- Logistic Regression
- Weight of Evidence (WoE) encoding
- Traditional credit risk feature engineering

This approach keeps the model transparent, explainable, and regulator friendly.

## What’s Inside

### Data Cleaning & Preprocessing
- Standardised inconsistent categorical values
- Handled missing delinquency data
- Cleaned noisy financial records

### Exploratory Data Analysis
- Analysed default behaviour across financial indicators
- Identified high-risk utilisation and debt patterns
- Explored relationships between borrower characteristics and default risk

### Feature Engineering
- Loan-to-income ratios
- Credit utilisation flags
- Inquiry behaviour patterns
- Log transformations for skewed variables
- Weight of Evidence encoding

### Machine Learning
- Logistic Regression model training
- Probability-based risk scoring
- Threshold tuning for lending decisions

### Business Dashboard
An interactive Dash dashboard allowing users to:
- Explore default trends
- Simulate approval thresholds
- Analyse portfolio risk
- Visualise business trade-offs

Example insight:
> Approve 67% of applicants with an estimated 9.8% bad rate.

## Tech Stack

### Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Dash
- Joblib

### Deployment
- Render
  
<img width="1339" height="568" alt="image" src="https://github.com/user-attachments/assets/85a0c851-8702-4f33-8bfc-106d9a6054a6" />


<img width="1001" height="492" alt="image" src="https://github.com/user-attachments/assets/1d212f10-2f00-43a1-8bd0-be5c4c2a0489" />


<img width="1345" height="636" alt="image" src="https://github.com/user-attachments/assets/ffb4a1aa-fe30-4fd5-afc5-624e86358304" />
