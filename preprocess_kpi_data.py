import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1.1 Load & Explore the Data
df = pd.read_csv("kpi_data.csv")
print(df.head())

# 1.2 Preprocess
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Select numerical features to normalize (excluding target and date)
features_to_normalize = [
    "active_users", "monthly_spend", "new_customers",
    "support_tickets", "login_count_avg"
]

scaler = MinMaxScaler()
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Save preprocessed data (optional)
df.to_csv("kpi_data_preprocessed.csv", index=False)
print(df.head())