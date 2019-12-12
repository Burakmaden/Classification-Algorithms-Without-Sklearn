import pandas as pd
import numpy as np
import random
from scipy.spatial import distance


def most_frequent(samples):
    counter = 0
    num = samples[0]

    for i in samples:
        curr_frequency = samples.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


class KNearestNeighbor:
    def __init__(self, neigh, t_size):
        self.neigh = neigh
        self.t_size = t_size

    def fit(self, fit_data):
        self.fit_data = fit_data

    def predict(self, testing_data):
        self.testing_data = testing_data
        prediction = []
        counter = 0

        # uzaklık bulma (öklid)
        for i in self.testing_data:
            classes = []
            distances = []
            for k in self.fit_data:
                x = np.asarray(i[:-1])
                y = np.asarray(k[:-1])
                dist = distance.euclidean(x, y)
                distances.append([dist, k[-1]])
            counter += 1
            distances.sort(key=lambda p: p[0])
            print("Test Count:{} || distances: {}" .format(counter, distances))
            # Sınıf tahminlerinin yapılması
            for l in range(0, self.neigh):
                classes.append(distances[l][1])
            guess = most_frequent(classes)
            prediction.append(guess)
            print("Prediction Class:{}".format(guess))

        # Finding Accuracy of Model
        len_predict = len(prediction)
        correct = 0
        wrong = 0
        for n in range(0, len_predict):
            if self.testing_data[n][4] == prediction[n]:
                correct += 1
            else:
                wrong += 1

        print("\nNumber of True Guesses:{} || Number of False Guesses:{}".format(correct, wrong))
        accuracy = correct / self.t_size
        print("Accuracy: {}%".format(round(accuracy*100, 2)))


# Main Blog
# Reading Dataset
df = pd.read_csv("iris.csv")
df.drop(['Id'], 1, inplace=True)
df.replace(['Iris-setosa'], -1, inplace=True)
df.replace(['Iris-versicolor'], 1, inplace=True)
df.replace(['Iris-virginica'], 2, inplace=True)
full_data = df.astype(float).values.tolist()
# Random State
random.seed(42)
random.shuffle(full_data)

# Test size and neighbors
test_size = 0.4
k_neigh = 5
test_data_size = test_size * len(full_data)
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

# Using knn
knn = KNearestNeighbor(k_neigh, test_data_size)
knn.fit(train_data)
knn.predict(test_data)