# cruzada1
import os
import warnings
import pandas as pd
import numpy  as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

datos = pd.read_csv("../../../datos/MuestraCredito5000V2.csv", delimiter = ';', decimal = ".")
datos.head()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
y = datos.loc[:, 'BuenPagador'].to_numpy()

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(sparse_output = False), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

mc_tt = []

for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)
  
  knn = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 8))
  ])
  noimprimir = knn.fit(X_train, y_train)
  MC = confusion_matrix(y_test, knn.predict(X_test))
  
  mc_tt.append(MC)
  
mc_tt

error_tt = []
for mc in mc_tt:
  error_tt.append(1 - (sum(mc.diagonal())/mc.sum()))

plt.figure(figsize = (15, 10))
plt.plot(error_tt, 'o-', lw = 2)
no_print = plt.xlabel("Número de Iteración", fontsize = 15)
no_print = plt.ylabel("Error Cometido %",    fontsize = 15)
no_print = plt.title("Variación del Error",  fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing'], loc = 'upper right', fontsize = 15)


mc_tc = []

for i in range(10):
  knn = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 8))
  ])
  noimprimir = knn.fit(X, y)
  MC = confusion_matrix(y, knn.predict(X))
  
  mc_tc.append(MC)

mc_tc
error_tc = []
for mc in mc_tc:
  error_tc.append(1 - (sum(mc.diagonal())/mc.sum()))

plt.figure(figsize=(12,8))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
no_print = plt.xlabel("Número de Iteración", fontsize = 15)
no_print = plt.ylabel("Error Cometido %", fontsize = 15)
no_print = plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa'], loc = 'upper right', fontsize = 15)

n = datos.shape[0]
mc_loo = []

for i in range(10):
  prediccion = []
  
  for j in range(n): 
    X_train = X.drop(j, axis = 0)
    X_test = X.iloc[j, :]
    y_train = np.delete(y, j)
    y_test = y[j]
    
    knn = Pipeline(steps=[
      ('preprocesador', preprocesamiento),
      ('clasificador', KNeighborsClassifier(n_neighbors = 8))
    ])
    noimprimir = knn.fit(X_train, y_train)
    pred_i = knn.predict(pd.DataFrame(X_test).T)
    
    prediccion.append(pred_i)
  
  MC = confusion_matrix(y, prediccion)
  mc_loo.append(MC)

mc_loo

error_loo = []
for mc in mc_loo:
  error_loo.append(1 - (sum(mc.diagonal())/mc.sum()))

plt.figure(figsize=(12,8))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.plot(error_loo, 'o-', lw = 2)
no_print = plt.xlabel("Número de Iteración", fontsize = 15)
no_print = plt.ylabel("Error Cometido %", fontsize = 15)
no_print = plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa', 'Dejando Uno Fuera'], loc = 'upper right', fontsize = 15)

#Enfoque K-fold 

#Tiene dos grandes problemas:

#La estimación del error tiende a ser muy variable dependiendo de cuáles datos quedan en la tabla #de training y cuáles en la tabla de testing.

#Se tiende a sobrestimar el error, es decir, es mucho mayor el error en la tabla de testing que en #toda la tabla de datos.

mc_cv = []

for i in range(10):
  kfold = KFold(n_splits = 5, shuffle = True)
  MC = 0
  
  for train, test in kfold.split(X, y):
    knn = Pipeline(steps=[
      ('preprocesador', preprocesamiento),
      ('clasificador', KNeighborsClassifier(n_neighbors = 8))
    ])
    noimprimir = knn.fit(X.iloc[train], y[train])
    pred_fold = knn.predict(X.iloc[test])
    MC = MC + confusion_matrix(y[test], pred_fold)
  
  mc_cv.append(MC)

mc_cv

error_cv = []
for mc in mc_cv:
  error_cv.append(1 - (sum(mc.diagonal())/mc.sum()))

plt.figure(figsize=(12, 8))
plt.plot(error_tt, 'o-', lw = 2)
plt.plot(error_tc, 'o-', lw = 2)
plt.plot(error_loo, 'o-', lw = 2)
plt.plot(error_cv, 'o-', lw = 2)
no_print = plt.xlabel("Número de Iteración", fontsize = 15)
no_print = plt.ylabel("Error Cometido", fontsize = 15)
no_print = plt.title("Variación del Error", fontsize = 20)
plt.grid(True)
plt.legend(['Training Testing', 'Tabla Completa', 'Dejando Uno Fuera', 'K-Fold CV'], loc = 'upper right', fontsize = 15)


Clasificacion 

datos = pd.read_csv("../../../datos/MuestraCredito5000V2.csv", delimiter = ';', decimal = ".")
datos.head()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(sparse_output = False), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

kfold = KFold(n_splits = 10, shuffle = True)
mc_sigmoid = 0
mc_rbf     = 0
mc_poly    = 0

for train, test in kfold.split(X, y):
  # sigmoid
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel = 'sigmoid', gamma = 'scale'))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_sigmoid = mc_sigmoid + confusion_matrix(y[test], pred_fold)
  
  # rbf
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel = 'rbf', gamma = 'scale'))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_rbf     = mc_rbf + confusion_matrix(y[test], pred_fold)
  
  # poly
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel = 'poly', gamma = 'scale'))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_poly    = mc_poly + confusion_matrix(y[test], pred_fold)

mc_sigmoid
mc_rbf
mc_poly

labels = ["Si", "No"]

indices_sigmoid = indices_general(mc_sigmoid, labels)
for k in indices_sigmoid:
  print("\n%s:\n%s" % (k, str(indices_sigmoid[k])))

labels = ["Si", "No"]

indices_rbf = indices_general(mc_rbf, labels)
for k in indices_rbf:
  print("\n%s:\n%s" % (k, str(indices_rbf[k])))

labels = ["Si", "No"]

indices_poly = indices_general(mc_poly, labels)
for k in indices_poly:
  print("\n%s:\n%s" % (k, str(indices_poly[k])))

p_global = pd.DataFrame({
  'Kernel' : ["sigmoid", "rbf", "poly"],
  'Valor'  : [indices_sigmoid["Precisión Global"], indices_rbf["Precisión Global"], indices_poly["Precisión Global"]]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_global)
no_print = plt.title("Precisión Global")
plt.show()

p_si = pd.DataFrame({
  'Kernel' : ["sigmoid", "rbf", "poly"],
  'Valor'  : [
    indices_sigmoid['Precisión por categoría']["Si"][0],
    indices_rbf['Precisión por categoría']["Si"][0],
    indices_poly['Precisión por categoría']["Si"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_si)
no_print = plt.title("Precisión Positiva")
plt.show()

p_no = pd.DataFrame({
  'Kernel' : ["sigmoid", "rbf", "poly"],
  'Valor'  : [
    indices_sigmoid['Precisión por categoría']["No"][0],
    indices_rbf['Precisión por categoría']["No"][0],
    indices_poly['Precisión por categoría']["No"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_no)
no_print = plt.title("Precisión Negativa")
plt.show()

#KNN

kfold = KFold(n_splits = 10, shuffle = True)
mc_ball_tree = 0
mc_kd_tree   = 0
mc_brute     = 0

for train, test in kfold.split(X, y):
  # ball_tree
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 5, algorithm = "ball_tree"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_ball_tree = mc_ball_tree + confusion_matrix(y[test], pred_fold)
  
  # kd_tree
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 5, algorithm = "kd_tree"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_kd_tree = mc_kd_tree + confusion_matrix(y[test], pred_fold)
  
  # brute
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 5, algorithm = "brute"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_brute   = mc_brute + confusion_matrix(y[test], pred_fold)

mc_ball_tree
mc_kd_tree
mc_brute

labels = ["Si", "No"]

indices_ball_tree = indices_general(mc_ball_tree, labels)
for k in indices_ball_tree:
  print("\n%s:\n%s" % (k, str(indices_ball_tree[k])))

labels = ["Si", "No"]

indices_kd_tree = indices_general(mc_kd_tree, labels)
for k in indices_kd_tree:
  print("\n%s:\n%s" % (k, str(indices_kd_tree[k])))


labels = ["Si", "No"]

indices_brute = indices_general(mc_brute, labels)
for k in indices_brute:
  print("\n%s:\n%s" % (k, str(indices_brute[k])))


p_global = pd.DataFrame({
  'Kernel' : ["ball_tree", "kd_tree", "brute"],
  'Valor'  : [
    indices_ball_tree["Precisión Global"],
    indices_kd_tree["Precisión Global"],
    indices_brute["Precisión Global"]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_global)
no_print = plt.title("Precisión Global")
plt.show()

p_si = pd.DataFrame({
  'Kernel' : ["ball_tree", "kd_tree", "brute"],
  'Valor'  : [
    indices_ball_tree['Precisión por categoría']["Si"][0],
    indices_kd_tree['Precisión por categoría']["Si"][0],
    indices_brute['Precisión por categoría']["Si"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_si)
no_print = plt.title("Precisión Positiva")
plt.show()

p_no = pd.DataFrame({
  'Kernel' : ["ball_tree", "kd_tree", "brute"],
  'Valor'  : [
    indices_ball_tree['Precisión por categoría']["No"][0],
    indices_kd_tree['Precisión por categoría']["No"][0],
    indices_brute['Precisión por categoría']["No"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_no)
no_print = plt.title("Precisión Negativa")
plt.show()

#bosques aleatorios

kfold = KFold(n_splits = 10, shuffle = True)
mc_gini     = 0
mc_entropy  = 0
mc_log_loss = 0

for train, test in kfold.split(X, y):
  # ball_tree
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 300, criterion = "gini"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_gini    = mc_gini + confusion_matrix(y[test], pred_fold)
  
  # kd_tree
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 300, criterion = "entropy"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_entropy = mc_entropy + confusion_matrix(y[test], pred_fold)
  
  # brute
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 300, criterion = "log_loss"))
  ])
  noimprimir  = modelo.fit(X.iloc[train], y[train])
  pred_fold   = modelo.predict(X.iloc[test])
  mc_log_loss = mc_log_loss + confusion_matrix(y[test], pred_fold)

mc_gini
mc_entropy
mc_log_loss


labels = ["Si", "No"]

indices_gini = indices_general(mc_gini, labels)
for k in indices_gini:
  print("\n%s:\n%s" % (k, str(indices_gini[k])))

labels = ["Si", "No"]

indices_entropy = indices_general(mc_entropy, labels)
for k in indices_entropy:
  print("\n%s:\n%s" % (k, str(indices_entropy[k])))

labels = ["Si", "No"]

indices_log_loss = indices_general(mc_log_loss, labels)
for k in indices_log_loss:
  print("\n%s:\n%s" % (k, str(indices_log_loss[k])))


p_global = pd.DataFrame({
  'Kernel' : ["gini", "entropy", "log_loss"],
  'Valor'  : [
    indices_gini["Precisión Global"],
    indices_entropy["Precisión Global"],
    indices_log_loss["Precisión Global"]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_global)
no_print = plt.title("Precisión Global")
plt.show()

p_si = pd.DataFrame({
  'Kernel' : ["gini", "entropy", "log_loss"],
  'Valor'  : [
    indices_gini['Precisión por categoría']["Si"][0],
    indices_entropy['Precisión por categoría']["Si"][0],
    indices_log_loss['Precisión por categoría']["Si"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_si)
no_print = plt.title("Precisión Positiva")
plt.show()

p_no = pd.DataFrame({
  'Kernel' : ["gini", "entropy", "log_loss"],
  'Valor'  : [
    indices_gini['Precisión por categoría']["No"][0],
    indices_entropy['Precisión por categoría']["No"][0],
    indices_log_loss['Precisión por categoría']["No"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_no)
no_print = plt.title("Precisión Negativa")
plt.show()

#todos los modelos


kfold = KFold(n_splits = 10, shuffle = True)
mc_knn = 0
mc_arbol = 0
mc_bosques = 0
mc_potenciacion = 0
mc_xg_potenciacion = 0
mc_svm = 0
mc_bayes = 0
mc_dis_lineal = 0
mc_dis_cuadratico = 0

for train, test in kfold.split(X, y):
  # KNN
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 5, algorithm = "ball_tree"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_knn     = mc_knn + confusion_matrix(y[test], pred_fold)
  
  # Arbol
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', DecisionTreeClassifier())
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_arbol   = mc_arbol + confusion_matrix(y[test], pred_fold)
  
  # Bosques Aleatorios
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 300, criterion = "entropy"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_bosques = mc_bosques + confusion_matrix(y[test], pred_fold)
  
  # Potenciación (Boosting)
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', AdaBoostClassifier(n_estimators = 10))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_potenciacion  = mc_potenciacion + confusion_matrix(y[test], pred_fold)
  
  # Potenciación Extrema (XGBoosting)
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', GradientBoostingClassifier(n_estimators=40))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_xg_potenciacion = mc_xg_potenciacion + confusion_matrix(y[test], pred_fold)
  
  # Máquinas de Soporte Vectorial
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel = 'rbf', gamma = 'scale'))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_svm     = mc_svm + confusion_matrix(y[test], pred_fold)
  
  # Método Ingenuo de Bayes
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', GaussianNB())
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_bayes   = mc_bayes + confusion_matrix(y[test], pred_fold)
  
  # Análisis Discriminte Lineal
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto'))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_dis_lineal  = mc_dis_lineal + confusion_matrix(y[test], pred_fold)
  
  # Análisis Discriminte Cuadrático
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', QuadraticDiscriminantAnalysis())
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  mc_dis_cuadratico = mc_dis_cuadratico + confusion_matrix(y[test], pred_fold)

mc_knn
mc_arbol
mc_bosques
mc_potenciacion
mc_xg_potenciacion
mc_svm
mc_bayes
mc_dis_lineal
mc_dis_cuadratico
labels = ["Si", "No"]

indices_knn = indices_general(mc_knn, labels)
for k in indices_knn:
  print("\n%s:\n%s" % (k, str(indices_knn[k])))
labels = ["Si", "No"]

indices_arbol = indices_general(mc_arbol, labels)
for k in indices_arbol:
  print("\n%s:\n%s" % (k, str(indices_arbol[k])))

labels = ["Si", "No"]

indices_bosques = indices_general(mc_bosques, labels)
for k in indices_bosques:
  print("\n%s:\n%s" % (k, str(indices_bosques[k])))

labels = ["Si", "No"]

indices_potenciacion = indices_general(mc_potenciacion, labels)
for k in indices_potenciacion:
  print("\n%s:\n%s" % (k, str(indices_potenciacion[k])))

labels = ["Si", "No"]

indices_xg_potenciacion = indices_general(mc_xg_potenciacion, labels)
for k in indices_xg_potenciacion:
  print("\n%s:\n%s" % (k, str(indices_xg_potenciacion[k])))

labels = ["Si", "No"]

indices_svm = indices_general(mc_svm, labels)
for k in indices_svm:
  print("\n%s:\n%s" % (k, str(indices_svm[k])))

labels = ["Si", "No"]

indices_bayes = indices_general(mc_bayes, labels)
for k in indices_bayes:
  print("\n%s:\n%s" % (k, str(indices_bayes[k])))

labels = ["Si", "No"]

indices_dis_lineal = indices_general(mc_dis_lineal, labels)
for k in indices_dis_lineal:
  print("\n%s:\n%s" % (k, str(indices_dis_lineal[k])))




labels = ["Si", "No"]

indices_dis_cuadratico = indices_general(mc_dis_cuadratico, labels)
for k in indices_dis_cuadratico:
  print("\n%s:\n%s" % (k, str(indices_dis_cuadratico[k])))

  
p_global = pd.DataFrame({
  'Kernel' : ["KNN", "Arbol", "Bosques", "ADA", "XGB", "SVM", "Bayes", "Lineal", "Cuadratico"],
  'Valor'  : [
    indices_knn["Precisión Global"],
    indices_arbol["Precisión Global"],
    indices_bosques["Precisión Global"],
    indices_potenciacion["Precisión Global"],
    indices_xg_potenciacion["Precisión Global"],
    indices_svm["Precisión Global"],
    indices_bayes["Precisión Global"],
    indices_dis_lineal["Precisión Global"],
    indices_dis_cuadratico["Precisión Global"]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_global)
no_print = plt.title("Precisión Global")
plt.show()



p_si = pd.DataFrame({
  'Kernel' : ["KNN", "Arbol", "Bosques", "ADA", "XGB", "SVM", "Bayes", "Lineal", "Cuadratico"],
  'Valor'  : [
    indices_knn['Precisión por categoría']["Si"][0],
    indices_arbol['Precisión por categoría']["Si"][0],
    indices_bosques['Precisión por categoría']["Si"][0],
    indices_potenciacion['Precisión por categoría']["Si"][0],
    indices_xg_potenciacion['Precisión por categoría']["Si"][0],
    indices_svm['Precisión por categoría']["Si"][0],
    indices_bayes['Precisión por categoría']["Si"][0],
    indices_dis_lineal['Precisión por categoría']["Si"][0],
    indices_dis_cuadratico['Precisión por categoría']["Si"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_si)
no_print = plt.title("Precisión Positiva")
plt.show()


p_no = pd.DataFrame({
  'Kernel' : ["KNN", "Arbol", "Bosques", "ADA", "XGB", "SVM", "Bayes", "Lineal", "Cuadratico"],
  'Valor'  : [
    indices_knn['Precisión por categoría']["No"][0],
    indices_arbol['Precisión por categoría']["No"][0],
    indices_bosques['Precisión por categoría']["No"][0],
    indices_potenciacion['Precisión por categoría']["No"][0],
    indices_xg_potenciacion['Precisión por categoría']["No"][0],
    indices_svm['Precisión por categoría']["No"][0],
    indices_bayes['Precisión por categoría']["No"][0],
    indices_dis_lineal['Precisión por categoría']["No"][0],
    indices_dis_cuadratico['Precisión por categoría']["No"][0]
  ]
})
fig, ax = plt.subplots(figsize = (12, 8))
no_print = sns.barplot(x = 'Kernel', y = 'Valor', data = p_no)
no_print = plt.title("Precisión Negativa")
plt.show()


#regresion

datos = pd.read_csv('../../../datos/boston_casas_v1.csv', delimiter = ',', decimal = ".", index_col = 0)

# Convierte las variables de object a categórica
datos['NegocMin'] = datos['NegocMin'].astype('category')
datos['LimitaRC'] = datos['LimitaRC'].astype('category')

datos.info()

datos = pd.read_csv('../../../datos/boston_casas_v1.csv', delimiter = ',', decimal = ".", index_col = 0)

# Convierte las variables de object a categórica
datos['NegocMin'] = datos['NegocMin'].astype('category')
datos['LimitaRC'] = datos['LimitaRC'].astype('category')

datos.info()
X = datos.loc[:, datos.columns != 'ValorProm']
y = datos.loc[:, 'ValorProm'].to_numpy()

#SVM
kfold = KFold(n_splits = 10, shuffle = True)
rmse_linear = 0
rmse_rbf    = 0
rmse_poly   = 0

for train, test in kfold.split(X, y):
  tam_test = y[test].size
  # sigmoid
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVR(kernel = 'linear', C=100, epsilon=0.1))
  ])
  noimprimir  = modelo.fit(X.iloc[train], y[train])
  pred_fold   = modelo.predict(X.iloc[test])
  rmse_linear = rmse_linear + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # rbf
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVR(kernel = 'rbf', C=100, epsilon=0.1))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_rbf   = rmse_rbf + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # poly
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVR(kernel = 'poly', C=100, epsilon=0.1))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_poly  = rmse_poly + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
# Se debe promediar el error cometido en los 10 grupos, 
# se divide entre 10 pues n_splits = 10
rmse_linear = rmse_linear / 10 
rmse_rbf = rmse_rbf / 10
rmse_poly = rmse_poly / 10

rmse_linear
rmse_rbf
rmse_poly

#Estructuramos en dataframe para el gráfico.
res = pd.DataFrame({
  'Kernel': ['linear', 'rbf', 'poly'],
  'RMSE': [rmse_linear, rmse_rbf, rmse_poly]
})
res

fig, ax = plt.subplots(figsize = (12, 8))
no_imprimir = sns.barplot(x = 'Kernel', y = 'RMSE', data = res)
no_imprimir = plt.title("Raíz de Error Cuadrático Medio")
plt.show()

#bosques aleatorios

kfold = KFold(n_splits = 10, shuffle = True)
rmse_squared_error  = 0
rmse_absolute_error = 0
rmse_friedman_mse   = 0
rmse_poisson        = 0

for train, test in kfold.split(X, y):
  tam_test = y[test].size
  
  # squared_error
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestRegressor(max_depth=2, criterion = "squared_error"))
  ])
  noimprimir  = modelo.fit(X.iloc[train], y[train])
  pred_fold   = modelo.predict(X.iloc[test])
  rmse_squared_error = rmse_squared_error + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # absolute_error
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestRegressor(max_depth=2, criterion = "absolute_error"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_absolute_error = rmse_absolute_error + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # friedman_mse
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestRegressor(max_depth=2, criterion = "friedman_mse"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_friedman_mse = rmse_friedman_mse + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # rmse_poisson
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestRegressor(max_depth=2, criterion = "poisson"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_poisson = rmse_poisson + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
# Se debe promediar el error cometido en los 10 grupos, 
# se divide entre 10 pues n_splits = 10  
rmse_squared_error = rmse_squared_error / 10
rmse_absolute_error = rmse_absolute_error / 10
rmse_friedman_mse = rmse_friedman_mse / 10
rmse_poisson = rmse_poisson / 10

rmse_squared_error
rmse_absolute_error
rmse_friedman_mse
rmse_poisson


res = pd.DataFrame({
  'Criterio': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
  'RMSE': [rmse_squared_error, rmse_absolute_error, rmse_friedman_mse, rmse_poisson]
})
res
fig, ax = plt.subplots(figsize = (12, 8))
no_imprimir = sns.barplot(x = 'Criterio', y = 'RMSE', data = res)
no_imprimir = plt.title("Raíz de Error Cuadrático Medio")
plt.show()

#Ridge 


kfold = KFold(n_splits = 10, shuffle = True)
rmse_svd       = 0
rmse_cholesky  = 0
rmse_lsqr      = 0
rmse_sparse_cg = 0
rmse_sag       = 0

for train, test in kfold.split(X, y):
  tam_test = y[test].size
  
  # svd
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "svd"))
  ])
  noimprimir  = modelo.fit(X.iloc[train], y[train])
  pred_fold   = modelo.predict(X.iloc[test])
  rmse_svd = rmse_svd + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # cholesky
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "cholesky"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_cholesky = rmse_cholesky + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # lsqr
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "lsqr"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_lsqr = rmse_lsqr + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # sparse_cg
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "sparse_cg"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_sparse_cg = rmse_sparse_cg + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # sag
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "sag"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_sag = rmse_sag + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))

# Se debe promediar el error cometido en los 10 grupos, 
# se divide entre 10 pues n_splits = 10    
rmse_svd = rmse_svd / 10
rmse_cholesky = rmse_cholesky / 10 
rmse_lsqr = rmse_lsqr / 10 
rmse_sparse_cg = rmse_sparse_cg / 10
rmse_sag = rmse_sag / 10

rmse_svd
rmse_cholesky
rmse_lsqr
rmse_sparse_cg
rmse_sag
#Estructuramos en dataframe para el gráfico.
res = pd.DataFrame({
  'Solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
  'RMSE': [rmse_svd, rmse_cholesky, rmse_lsqr, rmse_sparse_cg, rmse_sag]
})
res

fig, ax = plt.subplots(figsize = (12, 8))
sns.barplot(x = 'Solver', y = 'RMSE', data = res)
plt.title("Raíz de Error Cuadrático Medio")
plt.show()

#todos los modelos 

kfold = KFold(n_splits = 10, shuffle = True)
rmse_reg      = 0
rmse_lasso    = 0
rmse_ridge    = 0
rmse_svm      = 0
rmse_arboles  = 0
rmse_bosques  = 0
rmse_potencia = 0

for train, test in kfold.split(X, y):
  tam_test = y[test].size
  
  # Regresión
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', LinearRegression())
  ])
  noimprimir  = modelo.fit(X.iloc[train], y[train])
  pred_fold   = modelo.predict(X.iloc[test])
  rmse_reg = rmse_reg + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # Lasso
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Lasso(alpha = 0.1))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_lasso = rmse_lasso + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # Ridge
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', Ridge(alpha = 1.0, solver = "svd"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_ridge = rmse_ridge + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # SVM
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVR(kernel = 'rbf', C=100, epsilon=0.1))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_svm = rmse_svm + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # Árboles
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', DecisionTreeRegressor(max_depth = 3))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_arboles = rmse_arboles + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # Bosques Aleatorios
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestRegressor(max_depth=2, criterion = "poisson"))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_bosques = rmse_bosques + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
  
  # Potenciación
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', GradientBoostingRegressor(n_estimators = 300))
  ])
  noimprimir = modelo.fit(X.iloc[train], y[train])
  pred_fold  = modelo.predict(X.iloc[test])
  rmse_potencia = rmse_potencia + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))

# Se debe promediar el error cometido en los 10 grupos, 
# se divide entre 10 pues n_splits = 10   
rmse_reg = rmse_reg / 10
rmse_lasso = rmse_lasso / 10
rmse_ridge = rmse_ridge / 10
rmse_svm = rmse_svm / 10
rmse_arboles = rmse_arboles / 10
rmse_bosques = rmse_bosques / 10
rmse_potencia = rmse_potencia / 10


res = pd.DataFrame({
  'Modelo': ['Regresion', 'Lasso', 'Ridge', 'SVM', 'Arboles', 'Bosques', 'Potenciacion'],
  'RMSE': [rmse_reg, rmse_lasso, rmse_ridge, rmse_svm, rmse_arboles, rmse_bosques, rmse_potencia]
})
res

fig, ax = plt.subplots(figsize = (12, 8))
sns.barplot(x = 'Modelo', y = 'RMSE', data = res)
plt.title("Raíz de Error Cuadrático Medio")
plt.show()

#Se selecciona el mejor modelo, en este caso corresponde a Bosques Aleatorios, y con dicho modelo #se genera utilizando todos los datos.

modelo = Pipeline(steps=[
  ('preprocesador', preprocesamiento),
  ('clasificador', GradientBoostingRegressor(n_estimators = 300))
])
noimprimir = modelo.fit(X, y)

#Seguidamente se puede guardar el modelo en un archivo. Para ello, debemos apoyarnos del paquete #pickle.

import pickle

with open("modelo/modelo_potenciacion.pkl", 'wb') as file:  
  pickle.dump(modelo, file)

#Para cargar el modelo debemos realizar lo siguiente:

with open("modelo/modelo_potenciacion.pkl", 'rb') as file:  
  modelo = pickle.load(file)


#Finalmente podemos generar una predicción para unos datos nuevos.

datos_nuevos = pd.read_csv('../../../datos/boston_casas_nuevos_v1.csv', delimiter = ',', decimal = ".", index_col = 0)

datos_nuevos['NegocMin'] = datos_nuevos['NegocMin'].astype('category')
datos_nuevos['LimitaRC'] = datos_nuevos['LimitaRC'].astype('category')
datos_nuevos.head()

prediccion = modelo.predict(datos_nuevos)
prediccion


  





















