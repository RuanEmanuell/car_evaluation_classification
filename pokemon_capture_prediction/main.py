import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Lendo a base de dados
df = pd.read_csv("data/pokemon.csv")

# Visualizando primeiras linhas
print(df.head())

# Informações gerais
print(df.info())

# Removendo colunas que não ajudam muito
colunas_remover = ["name", "japanese_name", "classfication"]

for col in colunas_remover:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# # Preenche colunas numéricas com mediana (ex: se faltou altura = usa valor médio/mediano)
df.fillna(df.median(numeric_only=True), inplace=True)

# Preenche colunas texto com Unknown (ex: se faltou tipo = 'Unknown')
for col in df.select_dtypes(include="object").columns:
    df[col].fillna("Unknown")

# Convertendo colunas categóricas em números
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Variável alvo: capture_rate (taxa de captura)

y = df["capture_rate"]

# Features e treino
x = df.drop("capture_rate", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Modelo Linear Regression

lr_model = LinearRegression()

# Modelo Random Forest (com tuning)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_model = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=5,
    n_jobs=-1
)


# Predição de Linear Regression
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)

# Predição de Random Forest

rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)


print("\n===== LINEAR REGRESSION =====")
print("MAE :", lr_mae)
print("MSE :", lr_mse)
print("RMSE:", lr_rmse)

print("\n===== RANDOM FOREST =====")
print("MAE :", rf_mae)
print("MSE :", rf_mse)
print("RMSE:", rf_rmse)

results = {
    "Linear Regression": lr_rmse,
    "Random Forest": rf_rmse
}

best_model = min(results, key=results.get)

print("\nMelhor Modelo:", best_model)
print("Menor RMSE:", results[best_model])

# O melhor modelo foi escolhido com base no menor erro RMSE.
# O MAE mostra o erro médio absoluto.
# O MSE penaliza erros maiores.
# O RMSE facilita interpretação por estar na mesma escala do alvo.

# Linear Regression funciona melhor em relações lineares.
# Random Forest captura relações mais complexas e não lineares.

# Para este dataset, espera-se que Random Forest tenha desempenho superior.