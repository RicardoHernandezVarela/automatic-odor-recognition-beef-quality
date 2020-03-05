import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

#cargar el dataset en un DataFrame de Pandas
df = pd.read_csv('dataPCA.csv',delimiter=",")

print(df['class'].value_counts())

#Cambiar variables categoricas a numericas.
calidad = {'excellent':1, 'good':1, 'acceptable':1, 'spoiled':0}
df['class'] = [calidad[x] for x in df['class']]

#print(df.head())
#print(df['class'].value_counts())

#Separar el dataframe en features y target.
features = df.drop('class',axis=1).values
target = df['class'].values

#Importar train_test_split, dividir los datos en train y test.
from sklearn.model_selection import train_test_split

features_train,features_test,target_train,target_test = train_test_split(features,target,test_size=0.4,random_state=42, stratify=target)

#Definir y entrenanr el modelo de KNN.
from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors = 10)
KNN_mod.fit(features_train, target_train)

test = pd.DataFrame

#Evaluar el modelo.
test = pd.DataFrame(data = features_test,
              columns = ['muestra', 'principal component 1', 'principal component 2'])

print(test.head())
test['prediccion'] = KNN_mod.predict(features_test)
test['target'] = target_test
test['correcto'] = [1 if x == z else 0 for x, z in zip(test['prediccion'], target_test)]
accuracy = 100.0 * float(sum(test['correcto'])) / float(test.shape[0])
print(accuracy)

errores = test['prediccion'] == 0
print(errores.value_counts())
test.to_csv('test_results.csv')

#visualizar resultado de KNN
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Componente principal 1', fontsize = 15)
ax.set_ylabel('Componente principal 2', fontsize = 15)
ax.set_title('KNN, K = 10', fontsize = 20)

resultados = [1, 0]

colores = ['#3893e8', '#1c2345']

marcas = ["+", "o"]

for resultado, color, marca in zip(resultados, colores, marcas):
  indices = test['prediccion'] == resultado
  ax.scatter(test.loc[indices, 'principal component 1'], test.loc[indices, 'principal component 2'], c = color, s = 40, marker = marca)

ax.legend(resultados)
ax.grid()
plt.savefig('graph_KNN2.png')
plt.show()