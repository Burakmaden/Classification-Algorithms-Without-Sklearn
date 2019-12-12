import pandas as pd
import numpy as np
import math
import random


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector[:-1])
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers]) / float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        # print("clasValue:",classValue)
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilites = calculateClassProbabilities(summaries, inputVector)
    print(probabilites)
    bestLabel, bestProb = None, None
    for classValue, probability in probabilites.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        print(testSet[i][:-1], result)
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    df = pd.read_csv("iris.csv")
    df.drop(['Id'], 1, inplace=True)
    #df.drop(['PetalLengthCm'], 1, inplace=True)
    #df.drop(['PetalWidthCm'], 1, inplace=True)
    df.replace(['Iris-setosa'], 0, inplace=True)
    df.replace(['Iris-versicolor'], 1, inplace=True)
    df.replace(['Iris-virginica'], 2, inplace=True)
    #df.drop(df.index[100:], inplace=True)
    full_data = df.astype(float).values.tolist()
    random.seed(42)
    random.shuffle(full_data)

    test_size = 0.4
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    summaries = summarizeByClass(train_data)
    predictions = getPredictions(summaries, test_data)
    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy:{}%' .format(accuracy))

def heartmain():
    df = pd.read_csv('heart.csv')
    full_data = df.astype(float).values.tolist()
    random.seed(40)
    random.shuffle(full_data)
    test_size = 0.4
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    summaries = summarizeByClass(train_data)
    predictions = getPredictions(summaries, test_data)
    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy:{}%'.format(round(accuracy,2)))


# main()
heartmain()