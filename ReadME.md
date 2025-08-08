# Flask ML Churn Dashboard

A simple Flask web app that predicts customer churn risk using a TensorFlow model and displays recent business KPIs in a styled dashboard with a bar chart.

## Features

- Generates synthetic 60-day business KPI data
- Preprocesses and normalizes data
- Trains a TensorFlow model to predict churn
- Interactive dashboard with recent data and churn risk highlighting
- Bar chart visualization of churn risk

## Setup

1. **Clone or download this repo.**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate data:**
   ```bash
   python generate_kpi_data.py
   ```

4. **Preprocess data:**
   ```bash
   python preprocess_kpi_data.py
   ```

5. **Train the model:**
   ```bash
   python train_churn_model.py
   ```

6. **Run the Flask app:**
   ```bash
   python app.py
   ```

7. **Open your browser:**  
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure

```
flask-app/
│
├── app.py
├── generate_kpi_data.py
├── preprocess_kpi_data.py
├── train_churn_model.py
├── kpi_data.csv
├── kpi_data_preprocessed.csv
├── churn_model.h5
├── requirements.txt
└── templates/
    └── dashboard.html
```

## Notes

- Make sure your Python version is **3.7–3.11** for TensorFlow compatibility.
- The dashboard highlights churn risk and shows a bar chart for the last 10 days.
- For best results, balance your dataset if you want more positive churn predictions.

---
Made with Flask, Pandas, TensorFlow, and"# ML-Metrics-Hub-Live-KPIs-with-Predictive-Insights" 
