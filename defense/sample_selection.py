import pandas
import random


COUNT = 2000


samples = [[] for _ in range(10)]
explored = []

index = 0

while min([len(samples[k]) for k in range(10)]) < COUNT:
    df = pandas.read_csv("../white_box/mnist/mnist_simple_train/simple_image_" + str(index) + ".csv")
    np = df.to_numpy()

    label = int(df.columns[0][-6])

    for i in range(9):
        if np[1955, i] < 0.145:
            samples[label].append(np[:784, i].reshape((28, 28)))

    print([len(samples[k]) for k in range(10)])

    index += 1


print("done")
# [238, 1143, 2177, 1729, 1679, 1889, 1616, 1813, 1935, 1402]
# [1018, 1961, 1574, 1494, 1751, 1480, 1678, 1876, 1301, 1683]
