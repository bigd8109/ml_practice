from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn import preprocessing

def cal_gradient_0(theta_0, theta_1, xs, ys):
    # grad = np.uint64(0)
    grad = 0.0
    for i in range(len(xs)):
        grad = grad + (hypothesis(xs[i], theta_0, theta_1) - ys[i])
    return grad / (len(xs)) 

def cal_gradient_1(theta_0, theta_1, xs, ys):
    # grad = np.uint64(0)
    grad = 0.0
    for i in range(len(xs)):
        grad = grad + (hypothesis(xs[i], theta_0, theta_1) - ys[i])*xs[i]
    return grad / (len(xs))

def compute_cost(theta_0, theta_1, xs, ys):
    # cost = np.uint64(0)
    cost = 0.0
    for i in range(len(xs)):
        error = hypothesis(xs[i], theta_0, theta_1) - ys[i]
        cost = cost + error*error
    return cost / (2*len(xs))

def hypothesis(x, theta_0, theta_1):
    # return np.int64(theta_0 + x * theta_1)
    return theta_0 + x * theta_1

def main():
    print("Main")
    # Loading housing dataset
    (train_data, train_y), (test_data, test_y) = boston_housing.load_data()
    # print("TRaining size: ", train_data.shape, train_y.shape)
    xs1 = train_data[:,5]
    xs = xs1[0:100]
    # xs = preprocessing.normalize(train_data[:,5])
    # ys = preprocessing.normalize(train_y)
    ys = train_y[0:100]
    # print("Data size: ", xs.shape, ys.shape)
    # print(xs)
    # print (ys)
    # xs = [1, 3, 5, 3.25, 1.5]
    # ys = [1.8, 1.5, 2.25, 1.625, 1.0]
    threshold = 0.005
    alpha = 0.01

    # theta_0 = 2*random() - 1
    # theta_1 = 2*random() - 1
    theta_0 = 1
    theta_1 = 1

    gradient_0 = cal_gradient_0(theta_0, theta_1, xs, ys)
    gradient_1 = cal_gradient_1(theta_0, theta_1, xs, ys)
    iter = 1
    while(abs(gradient_0) > threshold and abs(gradient_1) > threshold):
        print("Iteration: ", iter) 
        print("Theta 0: {0:.10f}, Theta 1: {0:.10f}".format(theta_0, theta_1))
        print("Gradient 0: {0:.10f}, Gradient 1: {0:.10f}".format(gradient_0, gradient_1))
        print("Cost: {0:.10f}".format(compute_cost(theta_0, theta_1, xs, ys)))
        iter = iter + 1
        theta_0 = theta_0 - alpha * gradient_0
        theta_1 = theta_1 - alpha * gradient_1
        gradient_0 = cal_gradient_0(theta_0, theta_1, xs, ys)
        gradient_1 = cal_gradient_1(theta_0, theta_1, xs, ys)

        if (abs(gradient_0) > threshold):
            print("greater:", abs(gradient_0))        
    print("----------------------")
    print("FOUND THETA 0: {}, THETA 1: {} IS OPTIMUM".format(theta_0,theta_1))
    print("COST: {}".format(compute_cost(theta_0,theta_1, xs, ys)))

    x = np.linspace(0,10,100)

    # plt.subplot(1,2,2)
    plt.scatter(xs,ys,c="g")
    # plt.plot(xs,ys,c="g")
    plt.plot(x,theta_0 + theta_1*x, c="r")
    plt.title("Gradient descent")
    plt.show()

if __name__ == "__main__":
    main()
