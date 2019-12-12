import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # Train Data
        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        #Optimization
        step_size = [self.max_feature_value * 0.1,
                     self.max_feature_value * 0.01]

        # extremly expensive
        b_range_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_size:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple, step * b_range_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attemps to fix this a bit
                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], marker='*', c=self.colors[classification])
        return classification

    def visualize(self, x1, x2, title):
        [[self.ax.scatter(x[0], x[1], color=self.colors[i]) for x in train_set[i]] for i in train_set]

        # hyperplane x.w+b
        # v=x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        data_range = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'red', label='Sup. Vector 1')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'blue', label='Sup. Vector -1')

        # (w.x+b) = 0
        # decision boundry
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--', label='Hyperplane')

        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.title(title)
        plt.legend()
        plt.show()


df = pd.read_csv("iris.csv")
df.drop(['Id'], 1, inplace=True)
df.drop(['PetalLengthCm'], 1, inplace=True)
df.drop(['PetalWidthCm'], 1, inplace=True)
df.replace(['Iris-setosa'], -1, inplace=True)
df.replace(['Iris-versicolor'], 1, inplace=True)
df.drop(df.index[100:], inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.4
train_set = {-1: [], 1: []}
test_set = {-1: [], 1: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

svm = Support_Vector_Machine()
svm.fit(data=train_set)
t_data = [5.7, 2.9]
svm.predict(t_data)
x1 = 'SepalLengthCm'
x2 = 'SepalWidthCm'
title = 'Iris-Setosa vs Iris-Versicolor'
svm.visualize(x1, x2, title)
