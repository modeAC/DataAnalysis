from collections import defaultdict
from collections.abc import Iterable

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self):
        self.__class_frequency = defaultdict(lambda:0)
        self.__feat_frequency = defaultdict(lambda:0)

    def fit(self, X, y):
        for feature, label in zip(X, y):
            self.__class_frequency[label] += 1
            for value in feature:
                self.__feat_frequency[(value, label)] += 1

        num_samples = len(X)
        for k in self.__class_frequency:
            self.__class_frequency[k] /= num_samples

        for value, label in self.__feat_frequency:
            self.__feat_frequency[(value, label)] /= self.__class_frequency[label]

    def predict(self, X):
        predictions = []
        if isinstance(X, Iterable):
            for x in X:
                predictions.append(max(self.__class_frequency.keys(), key=lambda c: self.__calculate_class_freq(x, c)))
            return predictions
        elif isinstance(X, (int, float)):
            return max(self.__class_frequency.keys(), key=lambda c: self.__calculate_class_freq(X, c))

    def __calculate_class_freq(self, X, clss):
        freq = self.__class_frequency[clss]

        for feat in X:
            freq *= self.__feat_frequency.get((feat, clss), 10 ** (-6))
        return freq


if __name__ == '__main__':
    data = load_iris()

    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\nmodel accuracy = {accuracy_score(predictions, y_test)} (test_size=0.20)")

    model_sk = GaussianNB()
    model_sk.fit(X_train, y_train)
    predictions_sk = model_sk.predict(X_test)
    print(f"\nsklearn model accuracy = {accuracy_score(predictions_sk, y_test)}")

    # res = {}
    #
    # for p in range(1, 14):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p/20, random_state=42)
    #
    #     model = NaiveBayesClassifier()
    #     model.fit(X_train, y_train)
    #     predictions = model.predict(X_test)
    #     res[p/20] = accuracy_score(predictions, y_test)
    #
    # plt.plot(res.keys(), res.values())
    # plt.show()