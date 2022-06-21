import numpy as np
import requests
import csv
from matplotlib import pyplot as plt

class LinearRegression():
  def __init__(self, learning_rate: float=0.1):
    self.learning_rate = learning_rate
    np.random.seed(42)
    self.b_0 = np.random.randn(1)
    self.b_1 = np.random.randn(1)
    self.dataset_size = None

  def forward_prop(self, X):
    return self.b_0 * X + self.b_1

  def backward_prop(self, db_0, db_1):
    self.b_0 = self.b_0 - self.learning_rate * db_0
    self.b_1 = self.b_1 - self.learning_rate * db_1

  def gradient_descent(self, X, f):
    db_0 = (-2 * X * f).sum() / self.dataset_size
    db_1 = (-2 * f.sum()).sum() / self.dataset_size
    return db_0, db_1

  def loss_func(self, y, y_pred):
    return np.sum((y - y_pred)**2) / self.dataset_size

  def fit(self, X, y, epochs=10000):
    self.dataset_size = y.reshape((-1,)).shape[0]
    losses = []

    for i in range(epochs):
      y_pred = self.forward_prop(X)
      f = y - y_pred
      db_0, db_1 = self.gradient_descent(X, f)
      self.backward_prop(db_0, db_1)

      mse = self.loss_func(y,y_pred)
      losses.append(mse)
      # print(f"b_0 = {self.b_0}, b_1 = {self.b_1}, loss = {mse}")

    return losses

  def predict(self, X):
    y_pred = self.forward_prop(X)
    return y_pred


def csv2x_y(text):
  x = []
  y = []
  for line in (text).splitlines():
    res = line.split(",")
    x.append(float(res[1]))
    y.append(float(res[2]))
  return np.array(x), np.array(y)

if __name__ == '__main__':
    train_url = 'https://drive.google.com/file/d/1ZJ7USoXN5Iijv_j-o8igJKD6w5jr5QPF/view?usp=sharing'
    test_url = 'https://drive.google.com/file/d/1Dtf7FRjjhAGEikPFYYXmBBEdFFFWa-kK/view?usp=sharing'
    train_url = 'https://drive.google.com/uc?id=' + train_url.split('/')[-2]
    test_url = 'https://drive.google.com/uc?id=' + test_url.split('/')[-2]
    train = requests.get(train_url).text
    test = requests.get(test_url).text
    train = train.split("\n", 1)[1]
    test = test.split("\n", 1)[1]

    X_train, y_train = csv2x_y(train)
    X_test, y_test = csv2x_y(test)

    # print(f"X_train: {X_train[:5]}\ny_train: {y_train[:5]}")
    # print(f"\nX_test: {X_train[:5]}\ny_test: {y_train[:5]}")

    epochs = 10000
    lr = 0.01
    model = LinearRegression()
    losses = model.fit(X_train, y_train, epochs)
    # print(len(losses))
    # plt.plot(np.arange(epochs), losses)