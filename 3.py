import pandas
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pandas.read_csv("Datas/body.csv")
df = df.dropna(subset=["bmxleg", 'bmxwaist'])

plt.scatter(df['bmxleg'], df['bmxwaist'] , color="red")
plt.xlabel("Circunferencia Cintura (bmxwaist)")
plt.ylabel("Largo de Piernas (bmxleg)")
plt.show()