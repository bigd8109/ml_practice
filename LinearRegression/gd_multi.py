from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn import preprocessing

def cal_gradient(theta, xs, ys):
    grad = 0.0
    for i in range(len(xs)):
        grad = grad + (hypothesis(xs, theta) - ys[i])*xs[i]
    return grad / len(xs)

def compute_cost(theta, X, ys):
    cost = 0.0
    m, n = X.shape
    for i in range(m):
        error = hypothesis(X[i], theta) - ys[i]
        cost = cost + error*error
    return cost / (2*m)

def hypothesis(xs, theta):
    h = 0
    for i in range(len(xs)):
        h = h + xs[i] * theta[i]
    return h

def main():
    print("Main")
    # Loading housing dataset
    # (train_data, train_y), (test_data, test_y) = boston_housing.load_data()
    # print("TRaining size: ", train_data.shape, train_y.shape)
    X = np.array([[1, 1, 1], [1, 0, 1], [1, 2, -1]]) #x0 = 1
    ys = [4, 3, 1]
    # print("Data size: ", xs.shape, ys.shape)
    # print(xs)
    # print (ys)
    # xs = [1, 3, 5, 3.25, 1.5]
    # ys = [1.8, 1.5, 2.25, 1.625, 1.0]
    threshold = 0.005
    alpha = 0.01

    theta = [0.9, 0.8, 0.5]

    gradient = [0, 0, 0]
    for i in range(len(theta)):
        gradient[i] = cal_gradient(theta, X[i], ys)
    iter = 1
    while(abs(gradient[0]) > threshold and abs(gradient[1]) > threshold):
        print("Iteration: ", iter) 
        # print("Theta 0: {0:.10f}, Theta 1: {0:.10f}".format(theta_0, theta_1))
        # print("Gradient 0: {0:.10f}, Gradient 1: {0:.10f}".format(gradient_0, gradient_1))
        # print("Cost: {0:.10f}".format(compute_cost(theta_0, theta_1, xs, ys)))
        iter = iter + 1
        for i in range(len(theta)):
            print("Theta {} is {}".format(i, theta[i]))
            print("Gradient {} is {}".format(i, gradient[i]))
            print("Cost {} is {}".format(i, compute_cost(theta, X, ys)))
            theta[i] = theta[i] - alpha * gradient[i]
            gradient[i] = cal_gradient(theta, X[i], ys)
        # theta_0 = theta_0 - alpha * gradient_0
        # theta_1 = theta_1 - alpha * gradient_1
        # gradient_0 = cal_gradient_0(theta_0, theta_1, xs, ys)
        # gradient_1 = cal_gradient_1(theta_0, theta_1, xs, ys)
    print("----------------------")
    print("FOUND THETA 0: {}, THETA 1: {} , THETA 2: {} IS OPTIMUM".format(theta[0],theta[1], theta[2]))
    print("COST: {}".format(compute_cost(theta, X, ys)))

    # x = np.linspace(0,10,100)

    # # plt.subplot(1,2,2)
    # plt.scatter(xs,ys,c="g")
    # # plt.plot(xs,ys,c="g")
    # plt.plot(x,theta_0 + theta_1*x, c="r")
    # plt.title("Gradient descent")
    # plt.show()

if __name__ == "__main__":
    main()
