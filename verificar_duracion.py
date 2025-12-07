import pandas as pd

df = pd.read_csv("creditcard.csv")
max_time = df["Time"].max()
total_hours = max_time / 3600
print(f"Duración total del dataset: {total_hours:.2f} horas")