import random
from knn import KNN, dLabel

random.seed(1)
def loadData(filename):
    data = []
    with open(f"Datas/{filename}", "r") as f:
        lines = f.readlines()[1:]  # ignorar encabezado
        for line in lines:  # line: [x1,x2,x3,x4,clase]
            values = line.strip().split(",")
            clase = int(values.pop())
            features = list(map(float, values))
            data.append((features, clase))

    random.shuffle(data)
    total = len(data)
    split = int(total * 0.8) # 80% training, 20% test
    trainingData = data[:split]
    testData = data[split:]

    training = {}
    test = {}

    for features, clase in trainingData:
        if clase not in training:
            training[clase] = []
        training[clase].append(features)

    for features, clase in testData:
        if clase not in test:
            test[clase] = []
        test[clase].append(features)

    return training, test


training, test = loadData("data_training.txt")

aK = [2,3,4,5]

for nK in aK:
    total = 0
    correct = 0
    for clase, features in test.items():
        for f in features:
          total += 1
          result = KNN(training, f, nK)
          if result == dLabel[clase]:
            correct += 1
    print(f"Precision en K = {nK}: {correct/total * 100:.2f}%")