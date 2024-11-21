import pandas
# from sklearn.cluster import KMeans

df = pandas.read_csv("Datas/body.csv")
df = df[['bmxleg', 'bmxwaist']]


print(df)