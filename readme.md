# 📉 Telco Customer Churn Prediction – End-to-End MLOps Project

This repository contains an end-to-end machine learning project for predicting customer churn using Telco customer data. The project follows a production-oriented approach and covers the complete ML lifecycle, including data validation, model training, experiment tracking, API serving, containerization, CI/CD, and cloud deployment.

---

## 🎯 Problem Statement

Customer churn is a critical business problem. The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn based on demographic information, account details, and service usage.

---

## 📂 Dataset

- **Name:** Telco Customer Churn  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

**Target Variable:** `Churn` (Yes / No)

---

## 🧠 Machine Learning Pipeline

### 1. Data Validation
- Data quality checks using **Great Expectations**
- Schema validation, missing value checks, and consistency rules

### 2. Exploratory Data Analysis
- Churn distribution analysis
- Feature relationships and correlations

### 3. Data Preprocessing & Feature Engineering
- Encoding categorical features
- Scaling numerical features
- Train–test split

### 4. Model Training & Selection
- Multiple machine learning models trained and evaluated
- Hyperparameter tuning
- Best model selection using evaluation metrics

### 5. Experiment Tracking
- **MLflow** used to track:
  - Parameters
  - Metrics
  - Model artifacts
- Ensures reproducibility and comparison across experiments

---

## 📊 Model Evaluation

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 🌐 Model Serving

- Model served using **FastAPI**
- REST API accepts customer data as JSON
- Returns churn prediction with probability
- Designed for real-time inference

---

## 🐳 Containerization

- Application containerized using **Docker**
- Ensures environment consistency
- Simplifies deployment across platforms

---

## 🔄 CI/CD Pipeline

- **GitHub Actions** used for:
  - Running tests
  - Building Docker images
  - Automated deployment workflows

---

## ☁️ Cloud Deployment

- Deployed on **AWS ECS (Fargate)**
- **Application Load Balancer (ALB)** for traffic routing
- Scalable and production-ready architecture

---

## 🧰 Tech Stack

### Data & ML
- Python
- Pandas, NumPy
- Scikit-learn
- Great Expectations

### MLOps
- MLflow

### Backend
- FastAPI
- Uvicorn

### DevOps & Cloud
- Docker
- GitHub Actions
- AWS ECS (Fargate)
- AWS Application Load Balancer (ALB)

---
