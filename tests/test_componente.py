import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from datos import cargar_datos,ENTRADAS
from generar_dataset import generar
class DatosTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"datos.csv";self.df=generar(200)
    def tearDown(self):self.tmp.cleanup()
    def cargar(self,df):df.to_csv(self.path,index=False);return cargar_datos(self.path)
    def test_semilla_reproducible(self):pd.testing.assert_frame_equal(generar(200),generar(200))
    def test_semilla_distinta(self):self.assertFalse(generar(200,42).equals(generar(200,43)))
    def test_umbral_clases(self):
        d=generar(1000);e=np.where(d.pasajeros<=350,"baja",np.where(d.pasajeros<=800,"media","alta"));self.assertTrue((d.nivel_ocupacion==e).all())
    def test_duplicados(self):
        d=pd.concat([self.df,self.df.iloc[:1]],ignore_index=True);_,a=self.cargar(d);self.assertGreaterEqual(a["duplicados_eliminados"],1)
    def test_columna_faltante(self):
        with self.assertRaisesRegex(ValueError,"Faltan columnas"):self.cargar(self.df.drop(columns="hora"))
    def test_nulos(self):
        self.df.loc[0,"hora"]=float("nan")
        with self.assertRaisesRegex(ValueError,"faltantes"):self.cargar(self.df)
    def test_rango(self):
        self.df.loc[0,"hora"]=24
        with self.assertRaisesRegex(ValueError,"fuera de rango"):self.cargar(self.df)
    def test_etiqueta_incoherente(self):
        self.df.loc[0,"pasajeros"]=0;self.df.loc[0,"nivel_ocupacion"]="alta"
        with self.assertRaisesRegex(ValueError,"incoherente"):self.cargar(self.df)
    def test_sin_fuga_pasajeros(self):self.assertNotIn("pasajeros",ENTRADAS);self.assertNotIn("nivel_ocupacion",ENTRADAS)

from modelo import crear_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
class SupervisadoTest(unittest.TestCase):
    def test_categoria_nueva(self):
        df=generar(300);p=crear_pipeline(DecisionTreeClassifier(max_depth=5,random_state=42));p.fit(df[ENTRADAS],df.nivel_ocupacion)
        nuevo=df[ENTRADAS].iloc[:1].copy();nuevo["estacion"]="Estación no vista";self.assertIn(p.predict(nuevo)[0],["baja","media","alta"])
    def test_transformadores_independientes(self):
        a=crear_pipeline(DecisionTreeClassifier());b=crear_pipeline(DecisionTreeClassifier());self.assertIsNot(a["prep"],b["prep"])
    def test_particiones_disjuntas(self):
        d=generar(300);tr,te=train_test_split(d.index,test_size=.2,random_state=42,stratify=d.nivel_ocupacion);self.assertFalse(set(tr)&set(te));self.assertEqual(len(tr)+len(te),len(d))
