import math
dLabel = {0: "Iris-Setona", 1: "Versicolor", 2: "Virginica"}

def KNN(dData, aTest, k=3):
  aD = []
  for c in dData:
    for f in dData[c]:
      dE = math.sqrt(sum((a-b)**2 for a, b in zip(f,aTest))) # distancia euclidiana
      aD.append((dE, c))
  aD = sorted(aD[:k])
  for d in aD:
    pass

  return

with open("Datas/data_training.txt") as f:
  dD = {}
  for line in f:
    aL = line.split(",")
    print(aL)

aT = []

nK = 2
for aTest in aT:
  print(KNN(dD, aTest, nK))