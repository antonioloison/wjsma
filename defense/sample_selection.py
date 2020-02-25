import pandas
import random


COUNT = 2000


samples = [[] for _ in range(10)]
explored = []

while min([len(samples[k]) for k in range(10)]) < COUNT:
    index = random.randint(0, 15)

    if index in explored:
        continue

    explored.append(index)

    df = pandas.read_csv("basic_cleverhans_" + str(index) + ".csv")
    np = df.to_numpy()

    print(df.head())

    for i in range(9):
        if np[1955, i] < 0.145 and len(samples[i]) < COUNT:
            spl = int(df.columns[i][-1])

            samples[spl].append(np[:784, i].reshape((28, 28)))

    print([len(samples[k]) for k in range(10)])

print("done")