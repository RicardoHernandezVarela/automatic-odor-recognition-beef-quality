import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

#cargar el dataset en un DataFrame de Pandas
df = pd.read_csv('data.csv',delimiter=",")

from sklearn.preprocessing import StandardScaler

sensores = ['MQ135', 'MQ136', 'MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9']

#Separar los features
valores_sens = df.loc[:, sensores].values

#Separ los target
calidad = df.loc[:, ['class']].values

#Estandarizar los datos
valores_sens = StandardScaler().fit_transform(valores_sens)

#Aplicar PCA a los datos
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(valores_sens)
principalDf = pd.DataFrame(data = principalComponents,
              columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

#print(finalDf[0:5])
varianza_componentes = pca.explained_variance_ratio_

#visualizar resultado de PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Componente principal 1, var = {}%'.format(str(varianza_componentes[0]*100)), fontsize = 15)
ax.set_ylabel('Componente principal 2, var = {}%'.format(str(varianza_componentes[1]*100)), fontsize = 15)
ax.set_title('PCA 2 componentes', fontsize = 20)

targets = ['excellent', 'good', 'acceptable', 'spoiled']

colors = ['#3893e8', '#1c2345', '#ba160c', '#f04a00']

marks = ["p", "P", "*", "h"]

for target, color, mark in zip(targets, colors, marks):
  indicesToKeep = finalDf['class'] == target
  ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 40, marker = mark)

ax.legend(targets)
ax.grid()

plt.savefig('graph.png')
print("PCA finalizado")
print(finalDf['class'].value_counts())

finalDf.to_csv('dataPCA.csv')