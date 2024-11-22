import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv("Datas/body.csv")
df = df.dropna(subset=["bmxleg", 'bmxwaist'])

# Primera parte
plt.scatter(df['bmxleg'], df['bmxwaist'] , color="red")
plt.xlabel("Circunferencia Cintura (bmxwaist)")
plt.ylabel("Largo de Piernas (bmxleg)")
plt.show()

# Kmeans k=2
scaler = StandardScaler()
xScaled = scaler.fit_transform(df[["bmxleg", "bmxwaist"]])

kmeans2 = KMeans(2).fit(xScaled)

df['clusterK2'] = kmeans2.labels_
plt.scatter(df['bmxleg'], df['bmxwaist'], c=df['clusterK2'])
centroids2 = scaler.inverse_transform(kmeans2.cluster_centers_)
plt.scatter(centroids2[0:,0], centroids2[0:,1], color='red', s=100, label='Centroides')
plt.title('Tamaño Optimo Nuevos Bermudas para K = 2')
plt.xlabel('Largo de Piernas (bmxleg)')
plt.ylabel('Circunferencia Cintura (bmxwaist)')
plt.legend()
plt.grid(True)
plt.show()

# Las medidas de los nuevos bermudas son los centroides junma !!!

# Kmeans k=4
kmeans4 = KMeans(4).fit(xScaled)

df['clusterK4'] = kmeans4.labels_
plt.scatter(df['bmxleg'], df['bmxwaist'], c=df['clusterK4'])
centroids4 = scaler.inverse_transform(kmeans4.cluster_centers_)
plt.scatter(centroids4[0:,0], centroids4[0:,1], color='red', s=100, label='Centroides')
plt.title('Tamaño Optimo Nuevos Bermudas para K = 4')
plt.xlabel('Largo de Piernas (bmxleg)')
plt.ylabel('Circunferencia Cintura (bmxwaist)')
plt.legend()
plt.grid(True)
plt.show()