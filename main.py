import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Carregando os dados
colunas = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", names=colunas)

# Mostrando as 5 primeiras linhas do dataset
df.head()

# Mostrando algumas informações do Dataset
df.info()

# Pré-processamento (Encoding)
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Separando x e y
x = df.drop("class", axis=1)
y = df["class"]

# Modelos

# Decision Tree (com tuning)
dt_params = {
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

dt_model = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5
)


# KNN (com tuning)
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

knn_model = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5
)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

# Avaliação com 5-Fold Cross Validation dos 3 modelos, iniciando com Decision Tree
dt_scores = cross_val_score(dt_model, x, y, cv=5)
print("Decision Tree Scores:", dt_scores)
print("Decision Tree Mean Accuracy:", dt_scores.mean())
print()

# KNN
knn_scores = cross_val_score(knn_model, x, y, cv=5)
print("KNN Scores:", knn_scores)
print("KNN Mean Accuracy:", knn_scores.mean())
print()

# Logistic Regression
lr_scores = cross_val_score(lr_model, x, y, cv=5)
print("Logistic Regression Scores:", lr_scores)
print("Logistic Regression Mean Accuracy:", lr_scores.mean())
print()

# Armazenando os resultados para uma comparação final

results = {
    "Decision Tree": dt_scores.mean(),
    "KNN": knn_scores.mean(),
    "Logistic Regression": lr_scores.mean()
}

# Melhor modelo
best_model = max(results, key=results.get)

print("Best Model:", best_model)
print("Best Accuracy:", results[best_model])

# O melhor modelo foi selecionado com base na média da acurácia utilizando 5-Fold Cross Validation.
# Decision Tree geralmente performa bem neste dataset porque consegue lidar melhor com dados categóricos e criar regras de decisão baseadas nos atributos.
# KNN depende da distância entre os pontos e pode não ser tão eficiente com muitos dados categóricos.
# Logistic Regression pode ter desempenho inferior pois assume relações lineares entre as variáveis, o que nem sempre é adequado para este tipo de problema.

# Apesar de testar diferentes modelos como Decision Tree, KNN e Logistic Regression, a acurácia permaneceu em torno de 75–79%.
# Isso ocorre porque o dataset é altamente desbalanceado, com predominância da classe "unacc", fazendo com que os modelos priorizem essa classe.
