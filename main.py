import pandas as pd

colunas = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

df = pd.read_csv("data/car.data", names=colunas)

print(df.head())