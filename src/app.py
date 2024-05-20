from utils import db_connect
engine = db_connect()

# your code here
# Your code here
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. descargar data

url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
respuesta = requests.get(url)
nombre_archivo = "bank-marketing-campaign-data.csv"
with open(nombre_archivo, 'wb') as archivo:
     archivo.write(respuesta.content)

# 2. convertir csv en dataframe

total_data = pd.read_csv("../data/raw/bank-marketing-campaign-data.csv", sep=';')

# buscar duplicados
total_data_sin = total_data.drop_duplicates()  
#no duplicados

# Eliminar información obivamente irrelevante para el target
# Lo mas logico seria eliminar las siguentes columnas: 'contact', 'month', 'day_of_week', 'duration', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']
# tambien las columnas con alto numero de null: todas tienen el mismo numero entradas 

total_data.drop(['contact', 'month', 'day_of_week', 'duration', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed'], axis = 1, inplace = True)

# analisis de variable univariante : 8 variables categoricas y 3 variable numerica # Crear un histograma múltiple

fig, axis = plt.subplots(2, 4, figsize = (15, 7))
sns.histplot(ax = axis[0, 0], data = total_data, x = "job").set_xlim(-0.1, 1.1)
sns.histplot(ax = axis[0, 1], data = total_data, x = "marital").set(ylabel = None)
sns.histplot(ax = axis[0, 2], data = total_data, x = "education").set(ylabel = None)
sns.histplot(ax = axis[0, 3], data = total_data, x = "default")
sns.histplot(ax = axis[1, 0], data = total_data, x = "housing").set(ylabel = None)
sns.histplot(ax = axis[1, 1], data = total_data, x = "loan").set(ylabel = None)
sns.histplot(ax = axis[1, 2], data = total_data, x = "poutcome").set(ylabel = None)
sns.histplot(ax = axis[1, 3], data = total_data, x = "y").set(ylabel = None)
plt.tight_layout()
plt.show()

# Analisis univariable: 
# job: la gran mayoria de los entrevistados trabaja en servicios
# marital: el numero de casados es mucho mayor que el de solteros y divorciados juntos
# default: sobre 3/4 partes delos entrevistados tienen credito
# housing: hay poca diferencia enre usuariuos con prestamo de vivienda y sin el
# loan: solo 1/7 de los clientes tienen un prestamo personal 
# previous outcome: en la enorme mayoria de casos no habia registro del resultado de la campña anterior. ¿Nuevos clientes?
# target: aprox 1/7 de los clientes entrevistados aceptaron la oferta


# Analisis de variables numericas # Crear una figura múltiple con histogramas y diagramas de caja

fig, axis = plt.subplots(2, 3, figsize = (10, 7), gridspec_kw={'height_ratios': [6, 1]})

sns.histplot(ax = axis[0, 0], data = total_data, x = "campaign").set(xlabel = None)
sns.boxplot(ax = axis[1, 0], data = total_data, x = "campaign")
sns.histplot(ax = axis[0, 1], data = total_data, x = "age").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 1], data = total_data, x = "age")
sns.histplot(ax = axis[0, 2], data = total_data, x = "euribor3m").set(xlabel = None, ylabel = None)
sns.boxplot(ax = axis[1, 2], data = total_data, x = "euribor3m")
plt.tight_layout()
plt.show()

# variable target solo tiene respuestas yes/no asi que podemos convertirla en una variable numerica 1/0 
total_data['y_mapped'] = total_data['y'].map({'yes': 1, 'no': 0})

# Crear un diagrama de dispersión múltiple

fig, axis = plt.subplots(2, 2, figsize = (10, 7))
sns.regplot(ax = axis[0, 0], data = total_data, x = "age", y = "y_mapped")
sns.heatmap(total_data[["age", "y_mapped"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = total_data, x = "euribor3m", y = "y_mapped").set(ylabel=None)
sns.heatmap(total_data[["y_mapped", "euribor3m"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

plt.tight_layout()
plt.show()

# analisis de variables numericas
# La variable "age" tiene una distribucion asimetrica positiva y datos poco dispersos con sesgo hacia la derecha
# La variable "campaign" tiene muchos valores atipicos 
# la variable "euribor3m" tiene los datos muy dispersos y con distribucion muy asimetrica negativa 
# Ahora vamos a comparar la variable "age" y la "euribor3m" con la target "y"
# La variable target solo tiene respuestas yes/no asi que podemos convertirla en una variable numerica 1/0 


# De aqui se puede observar que la variable "age" tiene incidencia positiva muy debil en el target, 
# mientras que parece haber una relacion directa negativa entre la variable euribor3m y la contratacion del deposito (target). 
# Es decir, a mayor euribor menos numero de contrato de depositos.

# analisis categorico-categorico

fig, axis = plt.subplots(2, 3, figsize = (15, 7))

sns.countplot(ax = axis[0, 0], data = total_data, x = "job", hue = "y")
sns.countplot(ax = axis[0, 1], data = total_data, x = "marital", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[0, 2], data = total_data, x = "loan", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = total_data, x = "housing", hue = "y")
sns.countplot(ax = axis[1, 1], data = total_data, x = "education", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 2], data = total_data, x = "default", hue = "y").set(ylabel = None)
plt.tight_layout()
fig.delaxes(axis[1, 2])
plt.show()

# buscar valores unicos en education y job

grados_edu = total_data['education'].unique().tolist()
trabajos = total_data['job'].unique().tolist()

# Del gráfico anterior podemos obtener las siguientes conclusiones:

# mayor proporcion de solteros que de casados contrataron el deposito
# la misma proporcion de clientes con y sin prestamos de hogar contrataron el deposito >>> no afecta 
# similar proporcion de clientes con y sin prestamos personales contrataron el deposito >>> no afecta
# los grados de educacion son: ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'], asi que parece que la mayor proporcion de "yes" esta en "university.degree"
# los diferentes trabajos son "['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'], la mayor proporcion de contratos corresponde a "student"
# ******Segun este analisis previo los valores con mas alta proporcion de contratacion de depositos son: student - university degree - single 
# Usamos factorize para para codificar una variable categórica como una matriz de etiquetas numéricas y poder realizar el analisis de correlaciones necesitamos  

total_data["job_n"] = pd.factorize(total_data["job"])[0]
total_data["marital_n"] = pd.factorize(total_data["marital"])[0]
total_data["education_n"] = pd.factorize(total_data["education"])[0]
total_data["default_n"] = pd.factorize(total_data["default"])[0]
total_data["housing_n"] = pd.factorize(total_data["housing"])[0]
total_data["loan_n"] = pd.factorize(total_data["loan"])[0]

fig, axis = plt.subplots(figsize = (10, 6))
sns.heatmap(total_data[["job_n", "marital_n", "education_n", "default_n", "housing_n", "y_mapped", "euribor3m", "age", "campaign" ]].corr(), annot = True, fmt = ".2f")
plt.tight_layout()
plt.show()

# ingenieria de caracteristicas: analisis descriptivo

total_data.describe()

# Dibujar los diagramas de cajas de las variables 

fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = total_data, y = "age")
sns.boxplot(ax = axis[0, 1], data = total_data, y = "campaign")
sns.boxplot(ax = axis[0, 2], data = total_data, y = "euribor3m")
sns.boxplot(ax = axis[1, 0], data = total_data, y = "y_mapped")
sns.boxplot(ax = axis[1, 1], data = total_data, y = "job_n")
sns.boxplot(ax = axis[1, 2], data = total_data, y = "marital_n")
sns.boxplot(ax = axis[2, 0], data = total_data, y = "education_n")
sns.boxplot(ax = axis[2, 1], data = total_data, y = "default_n")
sns.boxplot(ax = axis[2, 2], data = total_data, y = "housing_n")

plt.tight_layout()
plt.show()

# las variables afectadas por outliers son: default_n, age, job_n, campaing, 
# Eliminamos: default_n, campaing
# Revisamos: age, job_n, 

# revisando el numero de outliers

age_stats = total_data["age"].describe()

# calculando upper y lower limit
age_iqr = age_stats["75%"] - age_stats["25%"]
upper_limit = age_stats["75%"] + 1.5 * age_iqr
lower_limit = age_stats["25%"] - 1.5 * age_iqr

print("estos son los upper y lower limites de edad:", upper_limit, "y", lower_limit)

total_data_fil = total_data[(total_data["age"] > 69.5)]
total_data_fil2 = total_data[(total_data["age"] < 9.5)]

# no hay valores nulos
total_data.isnull().sum().sort_values(ascending=False)

# - Outliers: (age) Solo 481 valores sobrepasan el upper limite y 18 el lower limit: 1.1678% de los registros. 
# - Análisis de valores faltantes: no hay valores nulos
# - SIGUIENTE PASO: Dividir el conjunto en train y test


# Dividir el conjunto en train y test 
# print(total_data.columns)

total_data = total_data.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'campaign', 'poutcome', 'y'], axis=1)

X = total_data.drop(['y_mapped'], axis=1) # Características (features)
y = total_data['y_mapped']  # Etiqueta (label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizacion escalar. 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index)

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index)

# feature selection
# Con un valor de k = 5, eliminar 2 características del conjunto de datos

from sklearn.feature_selection import f_classif, SelectKBest

selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

# Construye un modelo de regresión logística

# 1. entrenamiento del modelo


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced') # añadido parametro para 
model.fit(X_train_sel, y_train)

# 2. Prediccion del modelo

y_pred = model.predict(X_test_sel)
y_pred
y_pred.shape

# 3. optimizacion del modelo 
# test1 : The output indicates that your model is predicting all instances as the negative class (0), which is why you have high accuracy but precision, recall, and F1 score are all zero. 
# This is typically a sign of class imbalance or a model that is not well-tuned for the positive class.

# 4. metricas: accuracy, precision, recall, F1 score 
# test 2: añadido un parametro al modelo y otro a la metrica "precision"

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred, normalize=False)

precision = precision_score(y_test, y_pred, zero_division=0)  #añadido el parametro zero_division
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

# valores optimizados 
# - Accuracy: 71.67%
# - Precision: 24.25%
# - Recall: 70.48%
# - F1 Score: 36.09%
# - Confusion Matrix: [[5245, 2058], [276, 659]]

# Dibujaremos esta matriz para hacerla más visual

from sklearn.metrics import confusion_matrix

banking_cm = confusion_matrix(y_test, y_pred)
bank_df = pd.DataFrame(banking_cm)

plt.figure(figsize = (3, 3))
sns.heatmap(bank_df, annot=True, fmt="d", cbar=False)
plt.tight_layout()
plt.show()