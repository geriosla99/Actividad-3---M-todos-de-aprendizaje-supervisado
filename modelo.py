"""Aprendizaje supervisado: predicción del nivel de ocupación (baja/media/alta)
con Árbol de Decisión y comparación con Random Forest."""
import os
import matplotlib
matplotlib.use("Agg")
os.makedirs("docs", exist_ok=True)
os.makedirs("data", exist_ok=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1. Carga y exploración ------------------------------------------------------
df = pd.read_csv("dataset_transporte.csv")
print(df.info()); print(df.describe(include="all")); print(df.isna().sum())

# 2. Limpieza básica ----------------------------------------------------------
df = df.dropna().drop_duplicates()

# 3. Variables ----------------------------------------------------------------
# OJO: 'pasajeros' NO se usa como entrada porque la etiqueta se deriva de ella
# (data leakage). Solo se usa como entrada en un modelo de regresión.
cat = ["estacion", "linea"]
num = ["hora", "dia_semana", "es_festivo", "lluvia", "evento_especial"]
X, y = df[cat + num], df["nivel_ocupacion"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                         remainder="passthrough")

# 4. Modelos ------------------------------------------------------------------
modelos = {
    "Árbol de decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
resultados = {}
for nombre, clf in modelos.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cv = cross_val_score(pipe, X, y, cv=5).mean()
    resultados[nombre] = pipe
    print(f"\n=== {nombre} ===")
    print(f"Accuracy test: {accuracy_score(y_te, pred):.3f} | CV(5): {cv:.3f}")
    print(classification_report(y_te, pred))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, cmap="Blues")
    plt.title(nombre); plt.tight_layout()
    plt.savefig(f"docs/matriz_{'arbol' if 'rbol' in nombre else 'rf'}.png", dpi=150)
    plt.close()

# 5. Interpretación del árbol -------------------------------------------------
arbol = resultados["Árbol de decisión"]
nombres = arbol["prep"].get_feature_names_out()
clf = arbol["clf"]

print("\nReglas del árbol:\n", export_text(clf, feature_names=list(nombres), max_depth=4))
imp = pd.Series(clf.feature_importances_, index=nombres).sort_values(ascending=False)
print("\nImportancia de variables:\n", imp.head(8))

plt.figure(figsize=(22, 10))
plot_tree(clf, feature_names=nombres, class_names=clf.classes_,
          filled=True, max_depth=3, fontsize=8)
plt.savefig("docs/arbol_decision.png", dpi=150, bbox_inches="tight")
plt.close()

imp.head(8).plot(kind="barh").invert_yaxis()
plt.title("Importancia de variables"); plt.tight_layout()
plt.savefig("docs/importancia_variables.png", dpi=150)