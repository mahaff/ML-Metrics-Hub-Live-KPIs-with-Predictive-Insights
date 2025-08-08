from flask import Flask, render_template
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("churn_model.h5")
scaler = StandardScaler()

@app.route("/")
def dashboard():
    df = pd.read_csv("kpi_data.csv")
    df = df.sort_values("date", ascending=False).head(10)

    # Prepare features
    X = df[["active_users", "monthly_spend", "new_customers", "support_tickets", "login_count_avg"]]
    X_scaled = scaler.fit_transform(X)
    preds = model.predict(X_scaled).flatten()

    df["churn_risk"] = (preds > 0.5).astype(int)

    return render_template("dashboard.html", tables=[df.to_html(classes='data', index=False)], titles=df.columns.values)

if __name__ == "__main__":
    app.run(debug=True)
