# Customer Churn Predictor

An ML-powered web application that predicts whether a customer will leave a telecom company. Enter customer details and get instant churn prediction with probability score and feature importance chart.



##  What It Does

- Enter customer information like tenure, contract type, services used, and monthly charges
- Click **Predict Churn** 
- Get instant prediction with churn probability gauge
- See top 10 factors affecting churn decision
- Get retention suggestions if customer is likely to churn

---

## Features

-  Random Forest ML model trained on 7,000+ real customer records
-  Interactive churn probability gauge chart
-  Feature importance visualization
-  Retention suggestions for at-risk customers
-  Fast predictions using pre-trained model (.pkl file)
-  Clean 3-column Streamlit UI

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Model | Random Forest |
| Dataset | Telco Customer Churn |
| Training samples | ~5,600 |
| Test samples | ~1,400 |

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Scikit-learn | ML model training |
| Pandas | Data processing |
| Streamlit | Frontend web UI |
| Plotly | Interactive charts |
| Pickle | Model serialization |

---

##  Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Prashamsalohanii/customer-churn-predictor.git
cd customer-churn-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to https://kaggle.com/datasets/blastchar/telco-customer-churn
- Download and extract the zip file
- Copy WA_Fn-UseC_-Telco-Customer-Churn.csv into the project folder

### 4. Train the model
```bash
python model.py
```
This creates churn_model.pkl and feature_names.pkl

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Open browser
