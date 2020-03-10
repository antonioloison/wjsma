import numpy as np

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

import pandas as pd

first_index = 20
last_index = 30

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

import matplotlib.pyplot as plt

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[first_index:last_index]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[first_index:last_index]]))

# f, axarr = plt.subplots(10, 10)
# f.set_size_inches(16, 6)

# for i in range(first_index,last_index):
#     img = X_train[i]
#     axarr[i-first_index].imshow(img)
# plt.show()

indexes = {"airplane":30, "automobile": 4, "bird": 57, "cat":91, "deer":28, "dog":27, "frog":19, "horse": 7, "ship": 8, "truck":14}

#square
# for i, index in enumerate(indexes.values()):
#     file = r"D:\nn_robustness\cifar10_final\cifar10_simple_train" + "\\" + "simple_image_" + str(index) + ".csv"
#     csv = pd.read_csv(file)
#     for j in range(10):
#         if i == j:
#             img = csv["original_image_" + str(index)][:3072].values.reshape(32,32,3)
#         else:
#             img = csv["number_" + str(index) + "_" + str(i) + "_to_" + str(j)][:3072].values.reshape(32,32,3)
#         axarr[i,j].imshow(img)
#         axarr[i,j].axis('off')
# plt.show()


#one line
i = 1
f, axarr = plt.subplots(1, 10)
f.set_size_inches(16, 6)
file = r"C:\Users\Antonio\Projects\wjsma\visualization\first_mnist.csv"
csv = pd.read_csv(file)
for i in range(10):
    img = csv["image_" + str(i)][:784].values.reshape(28,28)
    axarr[i].imshow(img, 'gray')
    axarr[i].axis('off')
plt.show()

#two lines
# i = 1
# index = 4
# f, axarr = plt.subplots(2, 10)
# f.set_size_inches(16, 6)
# file_simple = r"D:\nn_robustness\cifar10_final\cifar10_simple_train" + "\\" + "simple_image_" + str(index) + ".csv"
# file_weighted = r"D:\nn_robustness\cifar10_final\cifar10_weighted_train" + "\\" + "weighted_image_" + str(index) + ".csv"
# csv_simple = pd.read_csv(file_simple)
# csv_weighted = pd.read_csv(file_weighted)
# for j in range(10):
#     if i == j:
#         img_simple = csv_simple["original_image_" + str(index)][:3072].values.reshape(32,32,3)
#         img_weighted = csv_weighted["original_image_" + str(index)][:3072].values.reshape(32,32,3)
#     else:
#         img_simple = csv_simple["number_" + str(index) + "_" + str(i) + "_to_" + str(j)][:3072].values.reshape(32,32,3)
#         img_weighted = csv_weighted["number_" + str(index) + "_" + str(i) + "_to_" + str(j)][:3072].values.reshape(32,32,3)
#     axarr[0,j].imshow(img_simple)
#     axarr[1,j].imshow(img_weighted)
#     axarr[0,j].axis('off')
#     axarr[1,j].axis('off')
# plt.show()

# Singl image
# index = 1
# f, axarr = plt.subplots(1, 3)
# f.set_size_inches(16, 6)
# file_simple = r"D:\nn_robustness\le_net_5_final\targeted\train\mnist_simple_train" + "\\" + "simple_image_" + str(index) + ".csv"
# file_weighted = r"D:\nn_robustness\le_net_5_final\targeted\train\mnist_weighted_train" + "\\" + "weighted_image_" + str(index) + ".csv"
# origin_class = 0
# target_class = 6
# for i in range(3):
#     if i == 0:
#         csv = pd.read_csv(file_simple)
#         img = csv["original_image_" + str(index)][:784].values.reshape(28,28)
#     elif i == 1:
#         csv = pd.read_csv(file_simple)
#         img = csv["number_" + str(index) + "_" + str(origin_class) + "_to_" + str(target_class)][:784].values.reshape(28,28)
#     elif i == 2:
#         csv = pd.read_csv(file_weighted)
#         img = csv["number_" + str(index) + "_" + str(origin_class) + "_to_" + str(target_class)][:784].values.reshape(28,28)
#     print(i)
#     axarr[i].imshow(img, 'gray')
#     axarr[i].axis('off')
# plt.show()
