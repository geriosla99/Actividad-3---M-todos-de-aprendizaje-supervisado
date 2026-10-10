"""Aprendizaje supervisado: demanda del Metro con Árbol de Decisión y Random Forest."""
from pathlib import Path
import argparse, json, platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from datos import cargar_datos, CATEGORICAS, NUMERICAS, ENTRADAS, CLASES
BASE = Path(__file__).resolve().parent

def crear_pipeline(clasificador):
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                              ("num", "passthrough", NUMERICAS)], remainder="drop")
    return Pipeline([("prep", prep), ("clf", clone(clasificador))])

def ejecutar(dataset=BASE/"dataset_transporte.csv", salida=BASE/"docs/resultados"):
    salida = Path(salida); salida.mkdir(parents=True, exist_ok=True)
    df, audit = cargar_datos(dataset)
    if set(df.nivel_ocupacion) != set(CLASES) or df.nivel_ocupacion.value_counts().min() < 10:
        raise ValueError("Se requieren las tres clases y al menos 10 ejemplos por clase")
    X, y = df[ENTRADAS], df.nivel_ocupacion
    tr, te = train_test_split(df.index, test_size=.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelos = {"base_mayoritaria": DummyClassifier(strategy="most_frequent"),
               "arbol": DecisionTreeClassifier(max_depth=5, random_state=42),
               "bosque": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2)}
    informe = {"auditoria": audit, "python": platform.python_version(), "sklearn": sklearn.__version__,
               "semilla":42, "entrenamiento":len(tr), "prueba":len(te), "entradas":ENTRADAS,
               "cv_solo_entrenamiento":True, "modelos":{}}
    pipelines = {}
    for nombre, estimador in modelos.items():
        pipe = crear_pipeline(estimador)
        scores = cross_validate(pipe, X.loc[tr], y.loc[tr], cv=cv,
                                scoring={"accuracy":"accuracy", "f1_macro":"f1_macro"})
        pipe.fit(X.loc[tr], y.loc[tr]); pred = pipe.predict(X.loc[te]); pipelines[nombre] = pipe
        informe["modelos"][nombre] = {"accuracy_test": accuracy_score(y.loc[te], pred),
              "f1_macro_test": f1_score(y.loc[te], pred, average="macro"),
              "cv_accuracy_media": float(scores["test_accuracy"].mean()),
              "cv_f1_macro_media": float(scores["test_f1_macro"].mean()),
              "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
              "cv_f1_folds": scores["test_f1_macro"].tolist(),
              "reporte":classification_report(y.loc[te], pred, labels=CLASES, output_dict=True, zero_division=0)}
        pd.DataFrame({"fila_limpia":te,"real":y.loc[te],"prediccion":pred}).to_csv(salida/f"predicciones_{nombre}.csv",index=False)
        ConfusionMatrixDisplay.from_predictions(y.loc[te], pred, labels=CLASES,cmap="Blues")
        plt.title(nombre.replace("_", " "));plt.tight_layout();plt.savefig(salida/f"matriz_{nombre}.png",dpi=150);plt.close()
    informe["seleccion_por_cv"] = max(["arbol","bosque"],key=lambda n:informe["modelos"][n]["cv_f1_macro_media"])
    tree = pipelines["arbol"]; names=tree["prep"].get_feature_names_out()
    (salida/"reglas_arbol.txt").write_text(export_text(tree["clf"],feature_names=list(names),max_depth=5),encoding="utf-8")
    plt.figure(figsize=(22,10));plot_tree(tree["clf"],feature_names=names,class_names=tree["clf"].classes_,filled=True,max_depth=3,fontsize=8)
    plt.title("Árbol de decisión: vista hasta profundidad 3; reglas completas en TXT")
    plt.savefig(salida/"arbol_decision.png",dpi=150,bbox_inches="tight");plt.close()
    imp=pd.Series(tree["clf"].feature_importances_,index=names).sort_values(ascending=False)
    imp.to_csv(salida/"importancia_variables.csv",header=["importancia"])
    imp.head(8).sort_values().plot.barh();plt.title("Importancia en el árbol (no causal)");plt.tight_layout();plt.savefig(salida/"importancia_variables.png",dpi=150);plt.close()
    pd.DataFrame({"fila_limpia":list(tr)+list(te),"particion":["entrenamiento"]*len(tr)+["prueba"]*len(te)}).to_csv(salida/"particiones.csv",index=False)
    (salida/"metricas.json").write_text(json.dumps(informe,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in informe.items() if k!="modelos"},ensure_ascii=False,indent=2))
    for n,r in informe["modelos"].items():print(f"{n}: accuracy={r['accuracy_test']:.4f}; F1 macro={r['f1_macro_test']:.4f}; CV F1={r['cv_f1_macro_media']:.4f}")
    return informe

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--datos",type=Path,default=BASE/"dataset_transporte.csv");p.add_argument("--salida",type=Path,default=BASE/"docs/resultados");a=p.parse_args();ejecutar(a.datos,a.salida)
