import pandas as pd
import numpy as np
from datetime import timedelta, date

np.random.seed(42)

start_date = date(2024, 1, 1)
days = 60

data = []
for i in range(days):
    d = start_date + timedelta(days=i)
    active_users = np.random.randint(800, 1500)
    monthly_spend = np.random.normal(24000, 2000)
    new_customers = np.random.randint(30, 90)
    support_tickets = np.random.randint(20, 60)
    login_count_avg = np.random.uniform(6, 10)
    churn_flag = 1 if active_users < 950 and support_tickets > 40 else 0

    data.append([d, active_users, monthly_spend, new_customers,
                 support_tickets, login_count_avg, churn_flag])

df = pd.DataFrame(data, columns=[
    "date", "active_users", "monthly_spend", "new_customers",
    "support_tickets", "login_count_avg", "churn_flag"
])

df.to_csv("kpi_data.csv", index=False)