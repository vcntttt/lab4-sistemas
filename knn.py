import math
dLabel = {0: "Iris-Setona", 1: "Versicolor", 2: "Virginica"}
# label = clase
def KNN(dData, aTest, k=3):
    aD = []
    for clase in dData:
        for feature in dData[clase]:
            # distancia euclidiana
            dE = math.sqrt(sum((a-b)**2 for a, b in zip(feature, aTest)))
            aD.append((dE, clase))
    aD.sort() # por defecto ordena por el primer elemento de la tupla asi que nos sirve

    vecinos = aD[:k]
    votes = {}
    for v in vecinos:
        clase = v[1] # segundo elemento de la tupla (clase)
        if clase in votes:
            votes[clase] += 1
        else:
            votes[clase] = 1

    result = max(votes, key=votes.get) # key -> para que sepa que valor comparar
    return dLabel[result]

def loadTrainingData(filename):
    data = {}
    with open(f"Datas/{filename}", "r") as f:
        lines = f.readlines()[1:]  # ignorar encabezado
        for line in lines:  # line: [x1,x2,x3,x4,clase]
            values = line.strip().split(",")
            clase = int(values.pop())
            features = list(map(float, values))
            if clase not in data:
                data[clase] = []
            data[clase].append(features)
    return data

'''
Data Training
{
  "0": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
}
'''

def loadTestData(filename):
    data = []
    with open(f"Datas/{filename}", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            values = line.strip().split(",")
            features = list(map(float, values))
            data.append(features)
    return data


dD = loadTrainingData("data_training.txt")
aT = loadTestData("data_test.txt")


# Ejecucion
aK = [2,3,4,5]

for nK in aK:
  print(f"Resultados para K = {nK}")
  for aTest in aT:
    print(KNN(dD, aTest, nK))
  print('\n')